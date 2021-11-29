#!/usr/bin/env node

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

const Path                        = require('path');
const wrtc                        = require('wrtc');
const Fastify                     = require('fastify');
const Peer                        = require('simple-peer');
const {renderLesson, renderEvent} = require('./render');

const fastify = Fastify()
                  .register(require('fastify-socket.io'))
                  .register(require('fastify-static'), {root: Path.join(__dirname, 'public')})
                  .get('/', (req, reply) => reply.sendFile('video.html'));

fastify.listen(8080)
  .then(() => fastify.io.on('connect', onConnect))
  .then(() => console.log('server ready'));

function onConnect(sock) {
  let cancelLesson = () => {};
  const peer            = new Peer({
    wrtc,
    sdpTransform: (sdp) => {
      // Remove bandwidth restrictions
      // https://github.com/webrtc/samples/blob/89f17a83ed299ef28d45b933419d809b93d41759/src/content/peerconnection/bandwidth/js/main.js#L240
      sdp = sdp.replace(/b=AS:.*\r\n/, '').replace(/b=TIAS:.*\r\n/, '');
      return sdp;
    }
  });
  peer.on('close', closeConnection);
  peer.on('error', closeConnection);
  peer.on('data', onDataChannelMessage);
  sock.on('disconnect', () => peer.destroy());
  sock.on('signal', (data) => { peer.signal(data); });
  peer.on('signal', (data) => { sock.emit('signal', data); });
  peer.on('connect', () => {
    const stream = new wrtc.MediaStream({id: `${sock.id}:video`});
    const source = new wrtc.nonstandard.RTCVideoSource({});
    stream.addTrack(source.createTrack());
    peer.addStream(stream);
    cancelLesson = renderLesson(
      {
        id: sock.id,
        width: 800,
        height: 600,
        lesson: '14',
        animationProps: {
          startTime: Date.now(),
          engineTime: 0,
          tick: 0,
          tock: 0,
          time: 0,
        }
      },
      source,
    );
  });

  function closeConnection(err) {
    console.log('connection closed' + (err ? ` (${err})` : ''));
    cancelLesson && cancelLesson({id: sock.id});
    cancelLesson = null;
    sock.disconnect(true);
    err ? peer.destroy(err) : peer.destroy();
  }

  function onDataChannelMessage(msg) {
    const {type, data} = (() => {
      try {
        return JSON.parse('' + msg);
      } catch (e) { return {}; }
    })();
    switch (data && type) {
      case 'event': {
        return renderEvent(sock.id, data);
      }
    }
  }
}
