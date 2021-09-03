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
const Peer = require('simple-peer');

function webrtcPlugin(fastify, options, done) {
  fastify.register(require('fastify-socket.io')).after(() => {
    fastify.io.on('connect', (sock) => {
      const peer = new Peer({wrtc, sdpTransform});
      sock.on('disconnect', () => peer.destroy());
      sock.on('signal', (data) => peer.signal(data));
      peer.on('signal', (data) => sock.emit('signal', data));
      peer.on('error', onError.bind(fastify, peer, sock.id, options.onError));
      peer.on('close', onClose.bind(fastify, peer, sock.id, options.onClose));
      peer.on('connect', onConnect.bind(fastify, peer, sock.id, options.onConnect));
    });
  });

  done();
}

function sdpTransform(sdp) {
  // Remove bandwidth restrictions
  // https://github.com/webrtc/samples/blob/89f17a83ed299ef28d45b933419d809b93d41759/src/content/peerconnection/bandwidth/js/main.js#L240
  sdp = sdp.replace(/b=AS:.*\r\n/, '').replace(/b=TIAS:.*\r\n/, '');
  return sdp;
}

function onError(peer, id, next, err) {
  this.log.warn({id, err}, `peer error`);
  if (typeof next === 'function') { next.call(this, peer, id, err); }
}

function onClose(peer, id, next) {
  this.log.debug({id}, `peer closed`);
  if (typeof next === 'function') { next.call(this, peer, id); }
}

function onConnect(peer, id, next) {
  this.log.debug({id}, `peer connected`);
  if (typeof next === 'function') { next.call(this, peer, id); }
}

module.exports = require('fastify-plugin')(webrtcPlugin);
