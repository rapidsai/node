// Copyright (c) 2021, NVIDIA CORPORATION.
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

const wrtc            = require('wrtc');
const {MemoryView}    = require('@rapidsai/cuda');
const {Float32Buffer} = require('@rapidsai/cuda');
const {GraphCOO}      = require('@rapidsai/cugraph');

const {loadNodes, loadEdges} = require('./loader');
const {RenderCluster}        = require('../../render/cluster');

const {create: shmCreate, detach: shmDetach} = require('shm-typed-array');

module.exports         = graphSSRClients;
module.exports.graphs  = Symbol('graphs');
module.exports.clients = Symbol('clients');

/**
 *
 * @param {import('fastify').FastifyInstance} fastify
 */
function graphSSRClients(fastify) {
  const graphs  = Object.create(null);
  const clients = Object.create(null);

  fastify.decorate(module.exports.graphs, graphs);
  fastify.decorate(module.exports.clients, clients);

  setInterval(layoutAndRenderGraphs(clients));

  return {onConnect, onData, onClose, onError: onClose};

  async function onConnect(sock, peer) {
    const {
      width      = 800,
      height     = 600,
      layout     = true,
      g: graphId = 'default',
    } = sock?.handshake?.query || {};

    const stream = new wrtc.MediaStream({id: `${sock.id}:video`});
    const source = new wrtc.nonstandard.RTCVideoSource({});

    clients[stream.id] = {
      video: source,
      state: {},
      event: {},
      props: {width, height, layout},
      graph: await loadGraph(graphId),
      frame: shmCreate(width * height * 3 / 2),
    };

    stream.addTrack(source.createTrack());
    peer.streams.push(stream);
    peer.addStream(stream);
  }

  function onData(sock, peer, message) {
    const [stream] = peer?.streams || [];
    if (stream && !peer.destroyed && !peer.destroying) {
      const {type, data} = (() => {
        try {
          return JSON.parse('' + message);
        } catch (e) { return {}; }
      })();
      switch (data && type) {
        case 'event': {
          clients[stream.id].event[data.type] = data;
          break;
        }
      }
    }
  }

  function onClose(sock, peer) {
    const [stream] = peer?.streams || [];
    if (stream) { delete clients[stream.id]; }
    const {g: graphId = 'default'} = sock?.handshake?.query || {};
    if (graphId in graphs) {
      if ((graphs[graphId].refCount -= 1) === 0) {  //
        delete graphs[graphId];
      }
    }
  }

  async function loadGraph(id) {
    if (!(id in graphs)) {
      const asDeviceMemory = (buf) => new (buf[Symbol.species])(buf);
      const [nodes, edges]         = await Promise.all([loadNodes(id), loadEdges(id)]);
      const src                    = edges.get('src');
      const dst                    = edges.get('dst');
      graphs[id]                   = {
        refCount: 0,
        nodes: {
          nodeRadius: asDeviceMemory(nodes.get('size').data),
          nodeFillColors: asDeviceMemory(nodes.get('color').data),
          nodeElementIndices: asDeviceMemory(nodes.get('id').data),
        },
        edges: {
          edgeList: asDeviceMemory(edges.get('edge').data),
          edgeColors: asDeviceMemory(edges.get('color').data),
          edgeBundles: asDeviceMemory(edges.get('bundle').data),
        },
        graph: new GraphCOO(src._col, dst._col, {directedEdges: true}),
      };
    }

    ++graphs[id].refCount;

    const pos = new Float32Buffer(Array.from(
      {length: graphs[id].graph.numNodes() * 2},
      () => Math.random() * 1000 * (Math.random() < 0.5 ? -1 : 1),
      ));

    return {
      gravity: 1.0,
      linLogMode: false,
      scalingRatio: 5.0,
      barnesHutTheta: 0.0,
      jitterTolerance: 0.05,
      strongGravityMode: false,
      outboundAttraction: false,
      graph: graphs[id].graph,
      edges: {
        ...graphs[id].edges,
        length: graphs[id].graph.numEdges(),
      },
      nodes: {
        ...graphs[id].nodes,
        length: graphs[id].graph.numNodes(),
        nodeXPositions: pos.subarray(0, pos.length / 2),
        nodeYPositions: pos.subarray(pos.length / 2),
      },
    };
  }
}

function layoutAndRenderGraphs(clients) {
  const renderer = new RenderCluster({numWorkers: 1 && 4});

  return () => {
    for (const id in clients) {
      const client = clients[id];

      if (client.isRendering) { continue; }

      const state = {...client.state};
      const props = {...client.props};
      const event =
        [
          'focus',
          'blur',
          'keydown',
          'keypress',
          'keyup',
          'mouseenter',
          'mousedown',
          'mousemove',
          'mouseup',
          'mouseleave',
          'wheel',
          'beforeunload'
        ].map((x) => client.event[x])
          .filter(Boolean);

      if (event.length === 0 && !props.layout) { continue; }
      if (event.length !== 0) { client.event = Object.create(null); }
      if (props.layout) { client.graph = forceAtlas2(client.graph); }

      const {
        width  = client.props.width ?? 800,
        height = client.props.height ?? 600,
      } = client.state;

      if (client.frame?.byteLength !== (width * height * 3 / 2)) {
        shmDetach(client.frame.key, true);
        client.frame = shmCreate(width * height * 3 / 2);
      }

      client.isRendering = true;

      renderer.render(id,
                      {
                        state,
                        props,
                        event,
                        frame: client.frame.key,
                        graph: {
                          ...client.graph,
                          graph: undefined,
                          edges: getIpcHandles(client.graph.edges),
                          nodes: getIpcHandles(client.graph.nodes),
                        },
                      },
                      (error, result) => {
                        client.isRendering = false;
                        if (id in clients) {
                          if (error) { throw error; }
                          result?.state && Object.assign(client.state, result.state);
                          // console.log(state?.deck?.props?.initialViewState);
                          // if (state?.deck?.props) {
                          //   client.props.initialViewState = state.deck.props.initialViewState;
                          // }
                          client.video.onFrame({...result.frame, data: client.frame.buffer});
                        }
                      });
    }
  }
}

function forceAtlas2({graph, nodes, edges, ...params}) {
  graph.forceAtlas2({...params, positions: nodes.nodeXPositions.buffer});
  return {
    graph,
    ...params,
    nodes: {...nodes, length: graph.numNodes()},
    edges: {...edges, length: graph.numEdges()},
  };
}

function getIpcHandles(obj) {
  const res = {};
  for (const key in obj) {
    const val = obj[key];
    res[key]  = val;
    if (val && (val instanceof MemoryView)) {  //
      try {
        res[key] = val.getIpcHandle().toJSON();
      } catch (e) {
        throw new Error([
          `Failed to get IPC handle for "${key}" buffer`,
          ...(e || '').toString().split('\n').map((x) => `\t${x}`)
        ].join('\n'));
      }
    }
  }
  return res;
}
