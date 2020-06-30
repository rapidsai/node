import rmm
import zmq
import cudf
import cupy
import numba
import cugraph
import traceback
import numpy as np
import pyarrow as pa

from cugraph.internals import GraphBasedDimRedCallback

edge_col_names = ['bundle', 'color', 'edge']
node_col_names = ['x', 'y', 'color', 'size', 'position']

def get_df(src, colnames):
    dst = cudf.DataFrame()
    for col in colnames:
        if col in src:
            dst[col] = src[col]
    return dst

def make_ipch(cols, name):
    if name in cols and len(cols[name]) > 0:
        hnd = sr_data_to_ipc_handle(cols[name])
        ary = np.array(hnd._ipc_handle.handle)
        return {'name': name, 'handle': hnd, 'ary': ary.tolist()}
    return None

def ipch_to_msg(tok):
    return {'name': tok['name'], 'data': tok['ary']}

def sr_data_to_device_ary(sr):
    size = sr.data.size // sr.dtype.itemsize
    return rmm.device_array_from_ptr(sr.data.ptr, size, dtype=sr.dtype)

def sr_data_to_ipc_handle(sr):
    return sr_data_to_device_ary(sr).get_ipc_handle()

def copy_cols(dst, src):
    for col in src.columns:
        if col in dst:
            cupy.copyto(
                cupy.asarray(sr_data_to_device_ary(dst[col])),
                cupy.asarray(sr_data_to_device_ary(src[col])))
        else:
            dst[col] = src[col].copy(True)
    return dst

def store_ipchs(dst, src):
    for col in src.keys():
        if col not in dst:
            dst[col] = make_ipch(src, col)
    return dst

def to_series_view(mem, dtype):
    dtype = np.dtype(dtype)
    mem = rmm.device_array_from_ptr(
        numba.cuda.cudadrv.driver.device_pointer(mem),
        mem.gpu_data.size // dtype.itemsize,
        dtype=np.uint8
    )
    return cudf.Series(cudf.core.Buffer(mem), dtype=dtype)

def create_node_positions_df(positions):
    return cudf.DataFrame.from_gpu_matrix(positions, columns=['x', 'y'])
    # return cudf.DataFrame({ 'position': to_series_view(positions, np.float32) })

class FA2ZmqCallback(GraphBasedDimRedCallback):

    def __init__(self, zmq_context, nodes, edges):
        super(FA2ZmqCallback, self).__init__()
        self._ctx = zmq_context
        self._nodes_df = nodes
        self._edges_df = edges
        self._edge_cols = dict()
        self._node_cols = dict()
        self._edge_ipch = dict()
        self._node_ipch = dict()
        self._pub_port = 6000
        self._cmd_port = self._pub_port + 1
        self._pub = self._ctx.socket(zmq.PUSH)
        self._cmd = self._ctx.socket(zmq.REP)
        self._pub.set_hwm(0)
        self._cmd.set_hwm(0)
        pub_url = 'tcp://{0}:{1}'.format('0.0.0.0', self._pub_port)
        cmd_url = 'tcp://{0}:{1}'.format('0.0.0.0', self._cmd_port)
        self._pub.bind(pub_url)
        self._cmd.bind(cmd_url)

    def on_preprocess_end(self, positions):
        try:
            self.connect()
            self.update(nodes=self._nodes_df, edges=self._edges_df)
        except Exception:
            print('on_preprocess_end err:', traceback.format_exc())

    def on_epoch_end(self, positions):
        try:
            self.update(nodes=create_node_positions_df(positions))
        except Exception:
            print('on_epoch_end err:', traceback.format_exc())

    def on_train_end(self, positions):
        try:
            print('on_train_end')
            self.update(nodes=create_node_positions_df(positions), msg=b'close')
            self.close()
        except Exception:
            print('on_train_end err:', traceback.format_exc())

    def connect(self):
        print('Waiting for subscriber to connect at tcp://{0}:{1}...'
             .format('0.0.0.0', self._pub_port))
        while self._cmd.recv() != b'ready':
            pass
        self._cmd.send(b'')
        return self

    def update(self, edges=None, nodes=None, msg=b''):
        if self._pub.closed:
            return self
        edges = get_df(cudf.DataFrame() if edges is None else edges, edge_col_names)
        nodes = get_df(cudf.DataFrame() if nodes is None else nodes, node_col_names)
        edge_ipchs = store_ipchs(self._edge_ipch, copy_cols(self._edge_cols, edges))
        node_ipchs = store_ipchs(self._node_ipch, copy_cols(self._node_cols, nodes))
        self._pub.send_json({
            'edge': list(map(ipch_to_msg, edge_ipchs.values())),
            'node': list(map(ipch_to_msg, node_ipchs.values())),
        })
        while self._cmd.recv() != b'ready':
            pass
        self._cmd.send(msg)
        self._edge_ipch.clear()
        self._node_ipch.clear()
        return self

    def close(self):
        self._pub.close()
        self._cmd.close()
        self._ctx.term()

    def _make_ipch(self, df):
        def _make_ipch(name):
            if name in df and len(df[name]) > 0:
                hnd = sr_data_to_ipc_handle(df[name])
                ary = np.array(hnd._ipc_handle.handle)
                return {'name': name, 'handle': hnd, 'ary': ary.tolist()}
            return None
        return _make_ipch

    def _ipch_to_msg(self):
        def tok_to_msg(tok):
            return {'name': tok['name'], 'data': tok['ary']}
        return tok_to_msg

nodes_df = cudf.DataFrame.from_arrow(
    pa.ipc.RecordBatchStreamReader(
        'data/01032018-webgl-nodes-small.arrow'
    ).read_all())

edges_df = cudf.DataFrame.from_arrow(
    pa.ipc.RecordBatchStreamReader(
        'data/01032018-webgl-edges-small.arrow'
    ).read_all())

graph_df = cudf.DataFrame.from_arrow(
    pa.ipc.RecordBatchStreamReader(
        'data/01032018-webgl-graph-small.arrow'
    ).read_all())

G = cugraph.Graph()
G.from_cudf_edgelist(graph_df, source='src', destination='dst', renumber=False)

nodes_df['size'] = (5 +
    ((G.in_degree()['degree'] + G.out_degree()['degree']).scale() * 250)
).astype(np.uint8)

edges_df['color'] = cudf.core.column.NumericalColumn(
    cudf.DataFrame({
        'src_color': graph_df[['src']].merge(nodes_df.rename(columns={'id': 'src'}, copy=False), on='src', how='left')['color'],
        'dst_color': graph_df[['dst']].merge(nodes_df.rename(columns={'id': 'dst'}, copy=False), on='dst', how='left')['color'],
    }).reset_index(drop=True).stack().reset_index(drop=True).data,
    dtype=np.uint32
)

cugraph.force_atlas2(
    G,
    max_iter=50,
    callback=FA2ZmqCallback(
        zmq.Context.instance(),
        nodes_df,
        edges_df
    )
)
