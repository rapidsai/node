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
const {RenderCluster} = require('@rapidsai/ssr-render-cluster');

const {create: shmCreate, detach: shmDetach} = require('shm-typed-array');

module.exports         = graphSSRClients;
module.exports.clients = Symbol('clients');

/**
 *
 * @param {import('fastify').FastifyInstance} fastify
 */
function graphSSRClients(fastify) {
  const clients = Object.create(null);

  fastify.decorate(module.exports.clients, clients);

  setInterval(layoutAndRenderPointCloud(clients));

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
      }
    }
  }

  function onClose(sock, peer) {
    const [stream] = peer?.streams || [];
    if (stream) { delete clients[stream.id]; }
  }
}

function layoutAndRenderPointCloud(clients) {
  const renderer = new RenderCluster(
    {numWorkers: 1 && 4, deckLayersPath: require('path').join(__dirname, 'deck')});

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
          'beforeunload',
          'shiftKey',
          'dragStart',
          'dragOver'
        ].map((x) => client.event[x])
          .filter(Boolean);

      if (event.length === 0 && !props.layout) { continue; }
      if (event.length !== 0) { client.event = Object.create(null); }

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

      renderer.render(id,
                      {
                        state,
                        props,
                        event,
                        frame: client.frame.key,
                      },
                      (error, result) => {
                        client.isRendering = false;
                        if (id in clients) {
                          if (error) { throw error; }

                          // copy result state to client's current state
                          result?.state && Object.assign(client.state, result.state);

                          client.video.onFrame({...result.frame, data: client.frame.buffer});
                        }
                      });
    }
  }
}
