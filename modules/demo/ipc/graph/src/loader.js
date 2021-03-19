// Copyright (c) 2020, NVIDIA CORPORATION.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import * as zmq from 'zeromq';
import { IpcMemory, Uint8Buffer } from '@nvidia/cuda';

import { pipe } from 'ix/asynciterable/pipe';
import { flatMap } from 'ix/asynciterable/operators/flatmap';
import { takeWhile } from 'ix/asynciterable/operators/takewhile';

const edgeBufferNames = new Set(['edge', 'color', 'bundle']);
const nodeBufferNames = new Set(['id', 'color', 'size', 'x', 'y']);

export default async function* loadGraphData(props = {}) {

  const ipcHandles = new zmq.Pull();
  const ipcRequests = new zmq.Request();
  const { protocol, hostname, port } = props.url;

  ipcHandles.connect(`${protocol}//${hostname}:${+port + 0}`);
  ipcRequests.connect(`${protocol}//${hostname}:${+port + 1}`);

  const next = () => request(ipcRequests, 'ready');
  let { numEdges = 0, numNodes = 0 } = await next();

  yield* pipe(
    ipcHandles,
    flatMap(async function* (buf) {
      const msg = tryJSONParse(buf);
      numEdges = Math.max(numEdges, msg.num_edges || 0);
      numNodes = Math.max(numNodes, msg.num_nodes || 0);
      const edges = openMemHandles(msg.edge, edgeBufferNames);
      const nodes = openMemHandles(msg.node, nodeBufferNames);
      const bbox = [msg.x_min, msg.x_max, msg.y_min, msg.y_max];
      const graph = createGraph(edges, nodes, numEdges, numNodes);
      const { promise, resolve: onAfterRender } = promiseSubject();
      yield { graph, bbox, onAfterRender };
      await promise.then(() => closeMemHandles([edges, nodes]));
      yield { graph, bbox };
    }),
    takeWhile(async (_, i) => i % 2 === 0 || ((await next()) !== 'close'))
  );

  [ipcHandles, ipcRequests].forEach(sock => sock.close());
}

async function request(sock, req = 'ready') {
  const resp = await sock.send(req)
    .then(() => sock.receive())
    .then((resp) => `${resp}`);
  switch (resp) {
    case '': case 'close': return resp;
    default: return tryJSONParse(resp);
  }
}

const createGraph = (edges, nodes, numEdges, numNodes) => ({
  numNodes,
  numEdges,
  nodeRadiusScale: 1 / 75,
  // nodeRadiusScale: 1/255,
  nodeRadiusMinPixels: 5,
  nodeRadiusMaxPixels: 150,
  data: (edges.size + nodes.size <= 0) ? {} : {
    edges: {
      offset: 0, length: numEdges, attributes: {
        edgeList: getBuffer(edges, 'edge', numEdges * 8),
        edgeColors: getBuffer(edges, 'color', numEdges * 8),
        edgeBundles: getBuffer(edges, 'bundle', numEdges * 8),
      }
    },
    nodes: {
      offset: 0, length: numNodes, attributes: {
        nodeRadius: getBuffer(nodes, 'size', numNodes * 1),
        nodeXPositions: getBuffer(nodes, 'x', numNodes * 4),
        nodeYPositions: getBuffer(nodes, 'y', numNodes * 4),
        nodeFillColors: getBuffer(nodes, 'color', numNodes * 4),
        nodeElementIndices: getBuffer(nodes, 'id', numNodes * 4),
      }
    },
  },
});

function getBuffer(map, key, size) {
  return map.has(key) ? map.get(key).subarray(0, size) : undefined;
}

function tryJSONParse(message = '') {
  try { return JSON.parse(message || '{}'); } catch (e) { return {}; }
}

function promiseSubject() {
  let resolve, reject;
  let promise = new Promise((r1, r2) => {
    resolve = r1;
    reject = r2;
  });
  return { promise, resolve, reject };
}

function openMemHandle(handle) {
  return new Uint8Buffer(new IpcMemory(handle));
}

function openMemHandles(handles, names) {
  return (Array.isArray(handles) ? handles : []).filter(Boolean)
    .filter(({ name, data = [] }) => data.length > 0 && names.has(name))
    .reduce((xs, { name, data }) => xs.set(name, openMemHandle(data)), new Map());
}

function closeMemHandle({ buffer }) {
  try { buffer.close(); } catch (e) { }
}

function closeMemHandles(maps) {
  return maps.forEach((handles) => handles.forEach(closeMemHandle));
}
