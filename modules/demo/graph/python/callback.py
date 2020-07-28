# Copyright (c) 2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rmm
import zmq
import cudf
import cupy
import asyncio
import cugraph
import traceback
import numpy as np

from cugraph.internals import GraphBasedDimRedCallback


class GraphZmqCallback(GraphBasedDimRedCallback):

    def __init__(
        self,
        zmq_ctx=None,
        zmq_port=6000,
        zmq_host='0.0.0.0',
        zmq_protocol='tcp://',
        edges=None, edge_col_names=['bundle', 'color', 'edge'],
        nodes=None, node_col_names=['x', 'y', 'color', 'size'],
        map_positions=lambda p: cudf.DataFrame.from_gpu_matrix(p, columns=['x', 'y'])
    ):
        super(GraphZmqCallback, self).__init__()
        self._ctx = zmq_ctx
        self._edge_cols = dict()
        self._node_cols = dict()
        self._edge_ipch = dict()
        self._node_ipch = dict()
        self._noop_df = cudf.DataFrame()
        self._map_positions = map_positions
        self._edge_col_names = list(edge_col_names)
        self._node_col_names = list(node_col_names)
        self._edges = self._noop_df if edges is None else edges
        self._nodes = self._noop_df if nodes is None else nodes
        self._num_nodes = len(self._nodes)
        self._num_edges = len(self._edges)
        self._pub_url = '{0}{1}:{2}'.format(zmq_protocol, zmq_host, zmq_port)
        self._cmd_url = '{0}{1}:{2}'.format(zmq_protocol, zmq_host, zmq_port + 1)
        self._connected = False

    def on_preprocess_end(self, positions):
        try:
            self.connect()
            edges = self._edges
            nodes = self._nodes.assign(**dict(
                self._map_positions(positions)[:self._num_nodes].iteritems()
            ))
            self._nodes, self._edges = (self._noop_df, self._noop_df)
            self.update(edges=edges, nodes=nodes, **self._get_bbox(nodes))
        except Exception:
            print('on_preprocess_end err:', traceback.format_exc())

    def on_epoch_end(self, positions):
        try:
            self.connect()
            nodes = self._map_positions(positions)[:self._num_nodes]
            self.update(nodes=nodes, **self._get_bbox(nodes))
        except Exception:
            print('on_epoch_end err:', traceback.format_exc())

    def on_train_end(self, positions):
        try:
            self.connect()
            nodes = self._map_positions(positions)[:self._num_nodes]
            self.update(nodes=nodes, **self._get_bbox(nodes))
        except Exception:
            print('on_train_end err:', traceback.format_exc())

    def connect(self):
        if not self._connected:
            self._pub = self._ctx.socket(zmq.PUSH)
            self._cmd = self._ctx.socket(zmq.REP)
            self._pub.set_hwm(0)
            self._cmd.set_hwm(0)
            self._pub.bind(self._pub_url)
            self._cmd.bind(self._cmd_url)
            print('Waiting for subscriber to connect at {0}...'.format(self._pub_url))
            while self._cmd.recv() != b'ready': pass
            self._connected = True
            self._cmd.send_json({
                'num_edges': self._num_edges,
                'num_nodes': self._num_nodes,
            })
        return self

    def update(self, edges=None, nodes=None, msg=b'', **kwargs):
        if not self._connected:
            return self
        edges = self._noop_df if edges is None else edges
        nodes = self._noop_df if nodes is None else nodes
        edges = self._filter(edges, self._edge_col_names)
        nodes = self._filter(nodes, self._node_col_names)
        edges = self._filter(self._copy(self._edge_cols, edges), edges.keys())
        nodes = self._filter(self._copy(self._node_cols, nodes), nodes.keys())
        edge_ipchs = self._store_ipchs(self._edge_ipch, edges)
        node_ipchs = self._store_ipchs(self._node_ipch, nodes)
        kwargs.update(edge=list(map(self._ipch_to_msg(), edge_ipchs.values())))
        kwargs.update(node=list(map(self._ipch_to_msg(), node_ipchs.values())))
        self._pub.send_json(kwargs)
        while self._cmd.recv() != b'ready': pass
        self._cmd.send(msg)
        self._edge_ipch.clear()
        self._node_ipch.clear()
        return self

    def close(self):
        if self._connected:
            self._pub.close()
            self._cmd.close()
            self._connected = False
            self._pub, self._cmd = (None, None)

    def _ipch_to_msg(self):
        def tok_to_msg(tok):
            return {'name': tok['name'], 'data': tok['ary']}
        return tok_to_msg

    def _sr_data_to_device_ary(self, sr):
        size = sr.data.size // sr.dtype.itemsize
        return rmm.device_array_from_ptr(sr.data.ptr, size, dtype=sr.dtype)

    def _sr_data_to_ipc_handle(self, sr):
        return self._sr_data_to_device_ary(sr).get_ipc_handle()

    def _filter(self, src, names):
        dst = dict()
        for col in names:
            if col in src:
                dst[col] = src[col]
        return dst

    def _copy(self, dst, src):
        for col in src.keys():
            if col in dst:
                dst_col = self._sr_data_to_device_ary(dst[col])
                src_col = self._sr_data_to_device_ary(src[col])
                cupy.copyto(cupy.asarray(dst_col), cupy.asarray(src_col))
            else:
                dst[col] = src[col].copy(True)
        return dst

    def _store_ipchs(self, dst, src):
        for col in src.keys():
            if col not in dst:
                dst[col] = self._make_ipch(src, col)
        return dst

    def _make_ipch(self, df, name):
        col = None if name not in df else df[name]
        if col is not None and len(col) > 0:
            hnd = self._sr_data_to_ipc_handle(col)
            ary = np.array(hnd._ipc_handle.handle)
            return {'name': name, 'handle': hnd, 'ary': ary.tolist()}
        return None

    def _get_bbox(self, positions):
        bbox = dict(x_min=0, x_max=0, y_min=0, y_max=0)
        bbox.update(num_edges=self._num_edges)
        bbox.update(num_nodes=self._num_nodes)
        if 'x' in positions:
            bbox.update(x_min=(0 + positions['x'].min()))
            bbox.update(x_max=(0 + positions['x'].max()))
        if 'y' in positions:
            bbox.update(y_min=(0 + positions['y'].min()))
            bbox.update(y_max=(0 + positions['y'].max()))
        return bbox

    def _blocking_wait(self, *tasks):
        tasks = asyncio.gather(*tasks)
        try:
            loop = asyncio.get_running_loop()
            loop.run_until_complete(tasks)
        except RuntimeError:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(tasks)
            loop.close()
