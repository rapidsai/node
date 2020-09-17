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

const wrtc = require('wrtc');
const Fastify = require('fastify');
const Peer = require('simple-peer');
const { createReactWindow } = require('@nvidia/glfw');

module.exports = function startServer({ url }) {

    const fastify = Fastify()
        .register(require('fastify-socket.io'))
        .register(require('fastify-static'), {
            root: require('path').join(__dirname, 'public')
        })
        .get('/', (req, reply) => reply.sendFile('video.html'));

    fastify.listen(8080)
           .then(() => fastify.io.on('connect', onConnect))
           .then(() => console.log('server ready'));

    function onConnect(sock) {

        const opts = {
            url,
            visible: !true,
            _animate: !true,
            transparent: false,
            serverRendered: true,
            // _forceNewWindow: true,
            _title: 'graph client',
        };
        
        const { open } = createReactWindow(`${__dirname}/src/app.js`, true);

        const controls = {
            stop() {},
            start() {},
            pause() {},
        };

        const peer = new Peer({
            wrtc,
            // Remove bandwidth restrictions
            // https://github.com/webrtc/samples/blob/89f17a83ed299ef28d45b933419d809b93d41759/src/content/peerconnection/bandwidth/js/main.js#L240
            sdpTransform: (sdp) => sdp.replace(/b=AS:.*\r\n/, '').replace(/b=TIAS:.*\r\n/, '')
        });
        peer.on('close', closeConnection);
        peer.on('error', closeConnection);
        peer.on('data', onDataChannelMessage);
        sock.on('disconnect', () => peer.destroy());
        sock.on('signal', (data) => peer.signal(data));
        peer.on('signal', (data) => sock.emit('signal', data));

        peer.on('connect', () => {
            open({ ...opts, _frames: createMediaStream(`${sock.id}:video`) });
        });

        function closeConnection(err) {
            console.log('closeConnection', err);
            sock.disconnect(true);
            err ? peer.destroy(err) : peer.destroy();
            if (window && !window.closed) {
                window.dispatchEvent({ type: 'close' });
            }
        }

        function createMediaStream(id) {
            const stream = new wrtc.MediaStream({ id, isScreencast: true });
            const source = new wrtc.nonstandard.RTCVideoSource({ isScreencast: true });
            stream.addTrack(source.createTrack());
            peer.addStream(stream);
            return source;
        }

        function onDataChannelMessage(data) {
            const msg = `${data}`;
            switch (msg) {
                case 'stop': return controls.stop();
                case 'play': return controls.start();
                case 'pause': return controls.pause();
                default:
                    try {
                        const { event } = JSON.parse(msg);
                        if (event && event.type) {
                            try {
                                window.dispatchEvent.call(global.window, event);
                            } catch (e) {
                                console.error(e && e.stack || e);
                            }
                        }
                    } catch (e) {}
                    break;
            }
        }
    }
}
