import os
import cupy
import numba
import numpy as np

import rmm
import zmq
import traceback
from cudf.core.index import RangeIndex
from cudf.core.column import CategoricalColumn
from cuml.internals import GraphBasedDimRedCallback

# GPU UMAP
import cudf
from cuml.manifold.umap import UMAP as cumlUMAP

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

if not os.path.exists('cuml/data/fashion'):
    print("error, data is missing!")

# https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py
def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

train, train_labels = load_mnist('cuml/data/fashion', kind='train')
test, test_labels = load_mnist('cuml/data/fashion', kind='t10k')
data = np.array(np.vstack([train, test]), dtype=np.float64) / 255.0
target = np.array(np.hstack([train_labels, test_labels]))

record_data = (('fea%d'%i, data[:,i]) for i in range(data.shape[1]))
gdf = cudf.DataFrame(dict(record_data))

classes = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot']

edge_col_names = ['color', 'edge', 'id']
node_col_names = ['position', 'size', 'color', 'edge_offset']

def filled_series(size, dtype, fill_value=0):
    sr = RangeIndex(0, size).to_series()
    sr[0:size] = fill_value
    return sr.astype(dtype)

def to_series_view(mem, dtype):
    dtype = np.dtype(dtype)
    mem = rmm.device_array_from_ptr(
        numba.cuda.cudadrv.driver.device_pointer(mem),
        mem.gpu_data.size // dtype.itemsize,
        dtype=dtype
    )
    return cudf.Series(cudf.core.Buffer(mem), dtype=dtype)

def create_initial_nodes_df(labels):
    color_indices = cudf.Series(labels.astype(np.int8))
    color_palette = cudf.Series([
        -12451426,-11583787,-12358156,-10375427,
        -7610114,-4194305,-6752794,-5972565,
        -5914010,-4356046,-6140066
    ])[:color_indices.max() + 1]
    dtype = cudf.core.dtypes.CategoricalDtype(ordered=True, categories=color_palette)
    color = CategoricalColumn(children=(color_indices._column,), dtype=dtype)
    color = cudf.Series(color.as_numerical_column(dtype=np.int32))
    return cudf.DataFrame({
        'color': color,
        'size': filled_series(len(color), 'int8', 20),
        'edge_offset': filled_series(len(color), 'int64', 0),
    })

def create_node_positions_df(embedding):
    return cudf.DataFrame({
        'position': to_series_view(embedding, np.float32) * 1000
    })

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

class UmapZmqCallback(GraphBasedDimRedCallback):

    def __init__(self, zmq_context):
        super(UmapZmqCallback, self).__init__()
        self._ctx = zmq_context
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

    def on_preprocess_end(self, embedding):
        try:
            self.connect()
            self.update(nodes=create_initial_nodes_df(target))
        except Exception:
            print('on_preprocess_end err:', traceback.format_exc())

    def on_epoch_end(self, embedding):
        try:
            self.update(nodes=create_node_positions_df(embedding))
        except Exception:
            print('on_epoch_end err:', traceback.format_exc())

    def on_train_end(self, embedding):
        try:
            self.update(nodes=create_node_positions_df(embedding))
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

    def update(self, edges=None, nodes=None):
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
        self._cmd.send(b'')
        # self._edge_ipch.clear()
        # self._node_ipch.clear()
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


def do_umap():
    cumlUMAP(
        n_neighbors=5,
        init="spectral",
        callback=UmapZmqCallback(zmq.Context.instance())
    ).fit_transform(gdf)

do_umap()
