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

import Loop from './luma';
import { CUDADevice, Uint8ClampedBuffer } from '@nvidia/cuda';
import { rgbaMirror, bgraToYCrCb420 } from '@nvidia/nvencoder';

const device = CUDADevice.new(1);

const fastify = Fastify()
    .register(require('fastify-socket.io'))
    .register(require('fastify-static'), {
        root: require('path').join(__dirname, 'static')
    })
    .get('/', (req, reply) => reply.sendFile('index.html'));

fastify.listen(8080).then(() => fastify.io.on('connect', onConnect));

function onConnect(sock) {

    let loop = undefined;
    const peer = new Peer({ wrtc });

    sock.on('disconnect', () => peer.destroy());
    sock.on('signal', (data) => peer.signal(data));
    peer.on('signal', (data) => sock.emit('signal', data));
    peer.on('close', () => { if (loop) { loop.stop(); } });
    peer.on('error', (err) => { if (loop) { loop.stop(); } });
    peer.on('connect', () => {
        if (loop) { loop.stop(); }

        Loop({ device }).then((opts) => {

            loop = opts.loop;
            let { width, height } = loop.animationProps;
            const stream = new wrtc.MediaStream({ id: 'video' });
            const source = new wrtc.nonstandard.RTCVideoSource();
            stream.addTrack(source.createTrack());
            peer.addStream(stream);

            let yuvDBuf = new Uint8ClampedBuffer(width * height * 3 / 2);
            let yuvHBuf = new Uint8ClampedArray(yuvDBuf.byteLength);

            opts.frames
                .on('error', (err) => {
                    peer.destroy(err);
                    sock.destroy(err);
                })
                .on('data', ({ width, height, data }) => {

                    if (yuvDBuf.byteLength !== (width * height * 3 / 2)) {
                        yuvDBuf = new Uint8ClampedBuffer(width * height * 3 / 2);
                        yuvHBuf = new Uint8ClampedArray(yuvDBuf.byteLength);
                    }

                    const resource = getRegisteredBufferResource(data);
                    CUDA.gl.mapResources([resource]);
                    const rgbaDBuf = new Uint8ClampedBuffer(CUDA.gl.getMappedPointer(resource));
                    // flip horizontally to account for WebGL's coordinate system (e.g. ffmpeg -vf vflip)
                    rgbaMirror(width, height, 0, rgbaDBuf);
                    // convert colorspace from OpenGL's RGBA to WebRTC's IYUV420
                    bgraToYCrCb420(yuvDBuf, rgbaDBuf, width, height);
                    CUDA.gl.unmapResources([resource]);
                    yuvDBuf.copyInto(yuvHBuf);
                    // Send the converted buffer to the client
                    source.onFrame({ width, height, data: yuvHBuf });
                });
        }).catch((err) => {
            peer.destroy(err);
            sock.destroy(err);
        });
    });

    peer.on('data', (data) => {
        const msg = `${data}`;
        switch (msg) {
            case 'stop': return loop && loop.stop();
            case 'play': return loop && loop.start();
            case 'pause': return loop && loop.pause();
            default:
                try {
                    const { event } = JSON.parse(msg);
                    if (event && event.type) {
                        window.dispatchEvent(event);
                    }
                } catch (e) {}
                break;
        }
    });
}

import { CUDA } from '@nvidia/cuda';

function getRegisteredBufferResource({ handle }) {
    if (handle && handle.cudaGraphicsResource === undefined) {
        handle.cudaGraphicsResource = CUDA.gl.registerBuffer(handle.ptr, 0);
    }
    return handle.cudaGraphicsResource;
}
