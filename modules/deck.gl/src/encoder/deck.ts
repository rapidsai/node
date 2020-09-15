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

import { PassThrough } from 'stream';
import { CUDADevice } from '@nvidia/cuda';
import { CUDAEncoderTransform } from '@nvidia/nvencoder';

export function createDeckGLVideoEncoderStream(deck: any, _device = CUDADevice.new(0)): Promise<NodeJS.ReadableStream> {

    const ffmpeg = require('fluent-ffmpeg');
    const { readPixelsToBuffer } = require('@luma.gl/webgl');

    let buffer: any = null;
    let buffers: PassThrough | null = null;
    let encoder: CUDAEncoderTransform | null = null;
    let outputs: NodeJS.ReadableStream | null = null;

    const {
        onResize: originalOnResize,
        onAfterRender: originalOnAfterRender,
        onWebGLInitialized: originalOnWebGLInitialized
    } = deck.props;

    const {
        onFinalize: originalOnFinalize,
    } = deck.animationLoop.props;

    let onDeckError = (_?: any) => {};
    let onDeckIntialized = (_?: any) => {};
    let onDeckIntializedPromise = new Promise((resolve, reject) => {
        onDeckIntialized = resolve;
        onDeckError = reject;
    });

    deck.props.onWebGLInitialized = function onWebGLInitialized(gl: WebGL2RenderingContext) {

        let result = undefined;

        try {
            if (originalOnWebGLInitialized) {
                result = originalOnWebGLInitialized.call(this, gl);
            }
        } catch (e) { return onDeckError(e); }

        if (!deck.props._framebuffer) {
            outputs = new PassThrough({ objectMode: true, highWaterMark: 1 });
            (outputs as any).end();
        } else {
            encoder = new CUDAEncoderTransform({
                width: gl.drawingBufferWidth,
                height: gl.drawingBufferHeight,
            });

            buffers = new PassThrough({ objectMode: true, highWaterMark: 1 });

            // Generates this ffmpeg command:
            // ffmpeg -i pipe:0 -f mp4 -vf vflip -movflags frag_keyframe+empty_moov pipe:1
            outputs = ffmpeg(buffers.pipe(encoder))
                .outputFormat('mp4')
                // flip vertically to account for WebGL's coordinate system
                .outputOptions('-vf', 'vflip')
                // create a fragmented MP4 stream
                // https://stackoverflow.com/a/55712208/3117331
                .outputOptions('-movflags', 'frag_keyframe+empty_moov');

            outputs!.once('end', () => deck.animationLoop._running && deck.finalize());
            outputs!.once('destroy', () => deck.animationLoop._running && deck.finalize());
        }

        window.addEventListener('close', () => deck.animationLoop._running && deck.finalize());

        onDeckIntialized();

        return result;
    };

    deck.props.onResize = function onResize(size: { width: number, height: number }) {
        if (encoder) { encoder.resize(size); }
        if (deck.props._framebuffer) {
            deck.props._framebuffer.resize(size);
        }
        if (originalOnResize) {
            originalOnResize.call(this, size);
        }
    };

    deck.props.onAfterRender = function onAfterRender(props: { gl: WebGL2RenderingContext }) {
        if (buffers && deck.animationLoop._running && deck.props._framebuffer) {
            const { gl } = props;
            buffers.write(buffer = readPixelsToBuffer(deck.props._framebuffer, {
                target: buffer, sourceType: gl.UNSIGNED_BYTE,
            }));
        }
        if (originalOnAfterRender) {
            originalOnAfterRender.call(this, props);
        }
    };

    deck.animationLoop.props.onFinalize = function onFinalize() {
        if (buffers) { buffers.end(); }
        if (encoder) { encoder.end(); }
        if (buffer) { buffer.delete({ deleteChildren: true }); }
        buffers = encoder = buffer = null;
        if (originalOnFinalize) {
            originalOnFinalize.apply(this, arguments);
        }
    };

    return onDeckIntializedPromise.then(() => outputs!);
}
