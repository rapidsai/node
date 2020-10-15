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
import { Device, Uint8Buffer } from '@nvidia/cuda';
import { rgbaMirror, bgraToYCrCb420 } from '@nvidia/nvencoder';

let DeckBuffer: any = undefined;
let Texture2D: any = undefined;
let Framebuffer: any = undefined;
let readPixelsToBuffer: any = undefined;

export function createDeckGLReactRef() {

    if (!DeckBuffer) {
        DeckBuffer = require(`../buffer`).Buffer;
    }

    if (!Texture2D || !Framebuffer || !readPixelsToBuffer) {
        ({ Texture2D, Framebuffer, readPixelsToBuffer } = require('@luma.gl/webgl'));
    }

    let _framebuffer: any = null;

    return {
        getRenderTarget() { return _framebuffer; },
        onWebGLInitialized(this: any /* Deck.props */, gl: WebGL2RenderingContext) {
            let result: any = undefined;
            if (this._originalOnWebGLInitialized) {
                result = this._originalOnWebGLInitialized.apply(this, arguments);
            }

            this._RGB_dbuffer = new DeckBuffer(gl, 0);
            this._YUV_dbuffer = new Uint8Buffer(0);
            this._YUV_hbuffer = new Uint8ClampedArray(0);

            _framebuffer = this._framebuffer || new Framebuffer(gl, {
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

            return result;
        },
        onResize(this: any /* Deck.props */, size: { width: number, height: number }) {
            if (_framebuffer) {
                _framebuffer.resize(size);
            }
            if (this._originalOnResize) {
                this._originalOnResize.apply(this, arguments);
            }
        },
        onBeforeRender(this: any /* Deck.props */) {
            if (this._originalOnAfterRender) {
                this._originalOnAfterRender.apply(this, arguments);
            }
        },
        onAfterRender(this: any /* Deck.props */, { gl }: { gl: WebGL2RenderingContext }) {
            if (this._originalOnAfterRender) {
                this._originalOnAfterRender.apply(this, arguments);
            }
            if (this._frames && _framebuffer) {

                const { width, height } = _framebuffer;
                // const width = gl.drawingBufferWidth;
                // const height = gl.drawingBufferHeight;

                const rgbByteLength = width * height * 4;
                const yuvByteLength = width * height * 3 / 2;

                if (this._RGB_dbuffer.byteLength !== rgbByteLength) {
                    this._RGB_dbuffer.delete({ deleteChildren: true });
                    this._RGB_dbuffer = new DeckBuffer(gl, {
                        byteLength: rgbByteLength,
                        accessor: { type: gl.UNSIGNED_BYTE, size: 4 }
                    });
                }

                if (this._YUV_dbuffer.byteLength !== yuvByteLength) {
                    this._YUV_dbuffer = new Uint8Buffer(yuvByteLength);
                    this._YUV_hbuffer = new Uint8ClampedArray(yuvByteLength);
                }

                const rgbGLBuffer = this._RGB_dbuffer;
                const yuvCUDABuffer = this._YUV_dbuffer;
                const yuvHOSTBuffer = this._YUV_hbuffer;

                // DtoD copy from framebuffer into pixelbuffer
                readPixelsToBuffer(_framebuffer, { target: rgbGLBuffer });
                // Map the GL buffer as a CUDA buffer for reading
                DeckBuffer.mapResources([rgbGLBuffer]);
                // Create a CUDA buffer view of the GL buffer
                const rgbCUDABuffer = rgbGLBuffer.asCUDABuffer();
                // flip horizontally to account for WebGL's coordinate system (e.g. ffmpeg -vf vflip)
                rgbaMirror(width, height, 0, rgbCUDABuffer);
                // convert colorspace from OpenGL's RGBA to WebRTC's IYUV420
                bgraToYCrCb420(yuvCUDABuffer, rgbCUDABuffer, width, height);
                // Unmap the GL buffer's CUDAGraphicsResource
                DeckBuffer.unmapResources([rgbGLBuffer]);
                // DtoH copy for output
                yuvCUDABuffer.copyInto(yuvHOSTBuffer);

                this._frames.onFrame({ width, height, data: yuvHOSTBuffer });
            }
        },
        // onFinalize(this: any /* AnimationLoop.props */) {
        //     if (this._originalOnFinalize) {
        //         return this._originalOnFinalize.apply(this.props, arguments);
        //     }
        //     const { deckProps } = this.props;
        //     if (deckProps) {
        //         this._deckProps = null;
        //         const { _frames, _RGB_dbuffer } = deckProps;
        //         if (_frames && _frames.end) {
        //             _frames.end();
        //         }
        //         if (_RGB_dbuffer) {
        //             _RGB_dbuffer.delete({ deleteChildren: true });
        //         }
        //     }
        // }
    };
}

export function createDeckGLVideoEncoderStream(
    // @ts-ignore
    deck: any,
    // @ts-ignore
    {
        device = new Device(0),
        // format = NvEncoderBufferFormat.ABGR
    }
): Promise<NodeJS.ReadWriteStream> {
    return Promise.resolve(new PassThrough());
}
