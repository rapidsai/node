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

module.exports = require('fastify-plugin')(function(fastify, opts, next) {
  fastify
    .register(require('fastify-socket.io'))  //
    .after(() => fastify.io.on('connect', onConnect));

  next();

  function onConnect(sock) {
    const peer = new Peer({
      wrtc,
      sdpTransform: (sdp) => {
        // Remove bandwidth restrictions
        // https://github.com/webrtc/samples/blob/89f17a83ed299ef28d45b933419d809b93d41759/src/content/peerconnection/bandwidth/js/main.js#L240
        sdp = sdp.replace(/b=AS:.*\r\n/, '').replace(/b=TIAS:.*\r\n/, '');
        return sdp;
      }
    });

    peer.on('close', onClose);
    peer.on('error', onError);
    peer.on('connect', onConnect);
    sock.on('disconnect', () => peer.destroy());

    // Handle signaling
    sock.on('signal', (data) => { peer.signal(data); });
    peer.on('signal', (data) => { sock.emit('signal', data); });

    if (typeof opts.onData === 'function') {
      peer.on('data', (message) => { opts.onData(sock, peer, message); });
    }

    const _onClose   = (typeof opts.onClose === 'function') ? opts.onClose : () => {};
    const _onError   = (typeof opts.onError === 'function') ? opts.onError : () => {};
    const _onConnect = (typeof opts.onConnect === 'function') ? opts.onConnect : () => {};

    function onClose() {
      fastify.log.info({id: sock.id}, `peer closed`);
      _onClose(sock, peer);
      sock.disconnect(true);
      peer.destroy();
    }

    function onError(err) {
      fastify.log.warn({id: sock.id, err}, `peer error`);
      _onError(sock, peer, err);
      sock.disconnect(true);
      peer.destroy(err);
    }

    function onConnect() {
      fastify.log.info({id: sock.id}, `peer connected`);
      _onConnect(sock, peer);
    }
  }
});
