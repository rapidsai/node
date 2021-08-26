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

const wrtc = require('wrtc');
const { nanoid } = require('nanoid');
const SimplePeer = require('simple-peer');

function webrtcPlugin(fastify, options, done) {
  const { prefix = '/api' } = options;
  const webrtcPeersSymbol = Symbol('simple-peers');

  fastify.decorate(webrtcPeersSymbol, {});

  fastify.decorate('getPeer', (id) => {
    return fastify[webrtcPeersSymbol][id];
  });

  fastify.decorate('delPeer', (id) => {
    console.log(`destroying peer ${id}`);
    const peer = fastify.getPeer(id);
    peer && peer.destroy();
    delete fastify[webrtcPeersSymbol][id];
  });

  fastify.decorate('newPeer', () => {
    const id = nanoid();
    console.log(`creating peer ${id}`);
    fastify[webrtcPeersSymbol][id] = new SimplePeer({ wrtc, sdpTransform })
      .on('close', fastify.delPeer.bind(fastify, id))
      .on('error', fastify.delPeer.bind(fastify, id))
      .on('connect', function onConnect() {  //
        console.log(`peer ${id} connected`);
      });
    return { id, peer: fastify[webrtcPeersSymbol][id] };
  });

  fastify.post(`${prefix}/rtc/signal`, (req, reply) => {
    const peer = fastify.getPeer(req.body.rtcId);
    if (!peer) {
      reply.status(401).send('Invalid rtcId');
    } else {
      peer.once('signal', handler).signal(req.body.offer);
      function handler(data) { reply.status(200).send(data); }
    }
  });

  done();
}

function sdpTransform(sdp) {
  // Remove bandwidth restrictions
  // https://github.com/webrtc/samples/blob/89f17a83ed299ef28d45b933419d809b93d41759/src/content/peerconnection/bandwidth/js/main.js#L240
  sdp = sdp.replace(/b=AS:.*\r\n/, '').replace(/b=TIAS:.*\r\n/, '');
  return sdp;
}

module.exports = require('fastify-plugin')(webrtcPlugin);
