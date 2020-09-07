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

export function videoEncoderCallbacks(_device = CUDADevice.new(0)) {

    const ffmpeg = require('fluent-ffmpeg');
    const { readPixelsToBuffer } = require('@luma.gl/webgl');
    const { Framebuffer, Texture2D } = require('@luma.gl/webgl');

    let buffer: any = null;
    /** @type {Framebuffer} */
    let framebuffer: any = null;
    let buffers: PassThrough | null = null;
    let encoder: CUDAEncoderTransform | null = null;

    return {
        _animate: true,
        createEncoderTarget(gl: WebGL2RenderingContext) {
            const size = {
                width: gl.drawingBufferWidth,
                height: gl.drawingBufferHeight,
            };

            encoder = new CUDAEncoderTransform(size);
            buffers = new PassThrough({ objectMode: true, highWaterMark: 1 });

            // Generates this ffmpeg command:
            // ffmpeg -i pipe:0 -f mp4 -vf vflip -movflags frag_keyframe+empty_moov pipe:1
            ffmpeg(buffers.pipe(encoder))
                .outputFormat('mp4')
                // flip vertically to account for WebGL's coordinate system
                .outputOptions('-vf', 'vflip')
                // create a fragmented MP4 stream
                // https://stackoverflow.com/a/55712208/3117331
                .outputOptions('-movflags', 'frag_keyframe+empty_moov')
                .pipe(process.stdout);

            framebuffer = new Framebuffer(gl, {
                color: new Texture2D(gl, {
                    mipmaps: false,
                    parameters: {
                        [gl.TEXTURE_MIN_FILTER]: gl.LINEAR,
                        [gl.TEXTURE_MAG_FILTER]: gl.LINEAR,
                        [gl.TEXTURE_WRAP_S]: gl.CLAMP_TO_EDGE,
                        [gl.TEXTURE_WRAP_T]: gl.CLAMP_TO_EDGE
                    }
                })
            });

            return framebuffer.resize(size) || framebuffer;
        },
        destroyEncoderTarget() {
            buffers && buffers.end();
            encoder && encoder.end();
            buffer && buffer.delete({ deleteChildren: true });
            framebuffer && framebuffer.delete({ deleteChildren: true });
            buffers = encoder = buffer = framebuffer = null;
        },
        onResize({ width, height }: { width: number, height: number }) {
            encoder && encoder.resize({ width, height });
            framebuffer && framebuffer.resize({ width, height });
        },
        onAfterRender({ gl }: { gl: WebGL2RenderingContext }) {
            buffers && buffers.write(buffer = readPixelsToBuffer(framebuffer, {
                target: buffer, sourceType: gl.UNSIGNED_BYTE,
            }));
        },
    }
}
