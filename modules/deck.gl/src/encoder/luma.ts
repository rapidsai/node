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

interface OnInitializeProps {
    gl: WebGL2RenderingContext;
    canvas: HTMLCanvasElement;
};

export function createAnimationLoopVideoEncoderStream(loop: any, _device = CUDADevice.new(0)): Promise<NodeJS.ReadableStream> {

    const ffmpeg = require('fluent-ffmpeg');
    const { readPixelsToBuffer } = require('@luma.gl/webgl');

    let buffer: any = null;
    let buffers: PassThrough | null = null;
    let encoder: CUDAEncoderTransform | null = null;
    let outputs: NodeJS.ReadableStream | null = null;

    const {
        onRender: originalOnRender,
        onFinalize: originalOnFinalize,
        onInitialize: originalOnInitialize,
    } = loop.props;

    let onAnimationLoopError = (_?: any) => {};
    let onAnimationLoopIntialized = (_?: any) => {};
    let onAnimationLoopIntializedPromise = new Promise((resolve, reject) => {
        onAnimationLoopIntialized = resolve;
        onAnimationLoopError = reject;
    });

    loop.props.onInitialize = function onInitialize(props: OnInitializeProps) {

        const { gl, canvas } = props;

        let result = undefined;
        
        try {
            if (originalOnInitialize) {
                result = originalOnInitialize.call(this, props);
            }
        } catch (e) { return onAnimationLoopError(e); }

        if (!loop.framebuffer) {
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

            canvas.addEventListener('resize', (_) => {
                const { width, height } = canvas;
                if (encoder) { encoder.resize({ width, height }); }
                if (loop.framebuffer) {
                    loop.framebuffer.resize({ width, height });
                }
            });

            outputs!.once('end', () => loop._running && loop.stop());
            outputs!.once('destroy', () => loop._running && loop.stop());
        }

        window.addEventListener('close', () => loop._running && loop.stop());

        onAnimationLoopIntialized();

        return result;
    };

    loop.props.onRender = function onRender(props: { gl: WebGL2RenderingContext }) {
        if (originalOnRender) {
            originalOnRender.call(this, props);
        }
        if (buffers && loop._running && loop.framebuffer) {
            const { gl } = props;
            buffers.write(buffer = readPixelsToBuffer(loop.framebuffer, {
                target: buffer, sourceType: gl.UNSIGNED_BYTE,
            }));
        }
    };

    loop.props.onFinalize = function onFinalize() {
        if (buffers) { buffers.end(); }
        if (encoder) { encoder.end(); }
        if (buffer) { buffer.delete({ deleteChildren: true }); }
        buffers = encoder = buffer = null;
        if (originalOnFinalize) {
            originalOnFinalize.apply(this, arguments);
        }
    };

    loop.start();

    return onAnimationLoopIntializedPromise.then(() => outputs!);
}
