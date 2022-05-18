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

const wrtc                                      = require('wrtc');
const {MemoryView, DeviceMemory, Float32Buffer} = require('@rapidsai/cuda');
const {DataFrame, Series, Float32}              = require('@rapidsai/cudf');
const {RenderCluster}                           = require('@rapidsai/ssr-render-cluster');

const {create: shmCreate, detach: shmDetach} = require('shm-typed-array');
const asDeviceMemory = (buf) => new (buf[Symbol.species])(buf);

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
      layout     = false,
      g: graphId = 'default',
    } = sock?.handshake?.query || {};

    const stream = new wrtc.MediaStream({id: `${sock.id}:video`});
    const source = new wrtc.nonstandard.RTCVideoSource({});

    clients[stream.id] = {
      video: source,
      state: {
        pickingMode: 'click',  // 'click', 'boxSelect'
        selectedInfo: {},
        boxSelectCoordinates: {rectdata: [{polygon: [[]], show: false}], startPos: null},
        clearSelections: false
      },
      event: {},
      props: {width, height, layout},
      graph: {},
      data: {
        nodes: {dataframe: new DataFrame({}), color: '', size: '', id: '', x: 'x', y: 'y'},
        edges: {dataframe: new DataFrame({}), color: '', id: '', bundle: '', src: 'src', dst: 'dst'}
      },
      frame: shmCreate(width * height * 3 / 2),
      peer: peer,
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
        case 'pickingMode': {
          clients[stream.id].state.pickingMode = data;
          break;
        }
        case 'clearSelections': {
          clients[stream.id].state.clearSelections = JSON.parse(data);
          break;
        }
        case 'layout': {
          clients[stream.id].props.layout = JSON.parse(data);
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
}

function layoutAndRenderGraphs(clients) {
  const renderer = new RenderCluster(
    {numWorkers: 1 && 4, deckLayersPath: require('path').join(__dirname, 'deck')});

  return () => {
    for (const id in clients) {
      const client = clients[id];

      if (client.isRendering) { continue; }

      const state = {...client.state};
      const props = {
        ...client.props,
        controller: (state.pickingMode === 'boxSelect' ? {dragPan: false} : {dragPan: true})
      };
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
          'beforeunload',
          'shiftKey',
          'dragStart',
          'dragOver'
        ].map((x) => client.event[x])
          .filter(Boolean);

      if (event.length === 0 && !props.layout) { continue; }
      if (event.length !== 0) { client.event = Object.create(null); }
      if (props.layout == true) { client.graph = forceAtlas2(client.graph); }

      const {
        width  = client.props.width ?? 800,
        height = client.props.height ?? 600,
      } = client.state;

      state.window = {width: width, height: height, ...client.state.window};

      if (client.frame?.byteLength !== (width * height * 3 / 2)) {
        shmDetach(client.frame.key, true);
        client.frame = shmCreate(width * height * 3 / 2);
      }
      client.isRendering = true;

      renderer.render(
        id,
        {
          state,
          props,
          event,
          frame: client.frame.key,
          layers: {
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
            if (client.state.clearSelections == true) {
              // clear selection is called once
              result.state.clearSelections = false;

              // reset selected state
              result.state.selectedInfo.nodes               = [];
              result.state.selectedInfo.edges               = [];
              result.state.selectedInfo.selectedCoordinates = {};
              result.state.boxSelectCoordinates.rectdata    = [{polygon: [[]], show: false}];

              // send to client
              client.peer.send(JSON.stringify({type: 'data', data: 'newQuery'}));
            } else if (JSON.stringify(client.state.selectedInfo.selectedCoordinates) !==
                       JSON.stringify(result.state.selectedInfo.selectedCoordinates)) {
              client.peer.send(JSON.stringify({type: 'data', data: 'newQuery'}));
            }
            // copy result state to client's current state
            result?.state && Object.assign(client.state, result.state);

            client.video.onFrame({...result.frame, data: client.frame.buffer});
          }
        });
    }
  }
}

function forceAtlas2({graph, nodes, edges, ...params}) {
  if (graph == undefined) { return {}; }
  const asDeviceMemory = (buf) => new (buf[Symbol.species])(buf);

  const positions = graph.forceAtlas2({...params, positions: nodes.nodeXPositions.data});

  nodes.nodeXPositions = asDeviceMemory(
    Series
      .new(
        {type: new Float32, length: graph.numNodes, offset: graph.numNodes, data: positions.buffer})
      .data);
  nodes.nodeYPositions = asDeviceMemory(
    Series.new({type: new Float32, length: graph.numNodes, offset: 0, data: positions.buffer})
      .data);

  return {
    graph: graph,
    ...params,
    nodes: {...nodes, length: graph.numNodes},
    edges: {...edges, length: graph.numEdges},
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
