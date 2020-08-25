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

const NVENCODER = (() => {
    let NVENCODER: any, types = ['Release'];
    if (process.env.NODE_DEBUG !== undefined || process.env.NODE_ENV === 'debug') {
        types.push('Debug');
    }
    for (let type; type = types.pop();) {
        try {
            if (NVENCODER = require(`../${type}/node_nvencoder.node`)) {
                break;
            }
        } catch (e) { console.error(e); continue; }
    }
    if (NVENCODER) return NVENCODER.init();
    throw new Error('node_nvencoder not found');
})();

export { NVENCODER };

export interface NVENCODER {
    NvEncoder: NvEncoderConstructor;
}

export interface NVEncoderOptions {

    // hevc: boolean;
    // device: number;
    width: number;
    height: number;

    // averageBitRate?: number;
    // maxBitRate?: number;
    // maxWidth?: number;
    // maxHeight?: number;
    // frameRateNumerator?: number;
    // frameRateDenomenator?: number;
}

export interface NvEncoderConstructor {
    readonly prototype: NvEncoder;
    new(options: NVEncoderOptions & { deviceType: number }): NvEncoder;
}

export interface NvEncoder {
    // width: number;
    // height: number;
}

export const NvEncoder: NvEncoderConstructor = NVENCODER.NvEncoder;

export enum NvEncoderDeviceType {
    DIRECTX = 0x0,
    CUDA    = 0x1,
    OPENGL  = 0x2,
}

export enum NvEncoderBufferFormat {
    /** Undefined buffer format */
    UNDEFINED                       = 0x00000000,
    /** Semi-Planar YUV [Y plane followed by interleaved UV plane] */
    NV12                            = 0x00000001,
    /** Planar YUV [Y plane followed by V and U planes] */
    YV12                            = 0x00000010,
    /** Planar YUV [Y plane followed by U and V planes] */
    IYUV                            = 0x00000100,
    /** Planar YUV [Y plane followed by U and V planes] */
    YUV444                          = 0x00001000,
    /**
     * 10 bit Semi-Planar YUV [Y plane followed by interleaved UV plane].
     * Each pixel of size 2 bytes. Most Significant 10 bits contain pixel data.
     */
    YUV420_10BIT                    = 0x00010000,
    /**
     * 10 bit Planar YUV444 [Y plane followed by U and V planes].
     * Each pixel of size 2 bytes. Most Significant 10 bits contain pixel data.
     */
    YUV444_10BIT                    = 0x00100000,
    /**
     * 8 bit Packed A8R8G8B8. This is a word-ordered format
     * where a pixel is represented by a 32-bit word with B
     * in the lowest 8 bits, G in the next 8 bits, R in the
     * 8 bits after that and A in the highest 8 bits.
     */
    ARGB                            = 0x01000000,
    /**
     * 10 bit Packed A2R10G10B10. This is a word-ordered format
     * where a pixel is represented by a 32-bit word with B
     * in the lowest 10 bits, G in the next 10 bits, R in the
     * 10 bits after that and A in the highest 2 bits.
     */
    ARGB10                          = 0x02000000,
    /**
     * 8 bit Packed A8Y8U8V8. This is a word-ordered format
     * where a pixel is represented by a 32-bit word with V
     * in the lowest 8 bits, U in the next 8 bits, Y in the
     * 8 bits after that and A in the highest 8 bits.
     */
    AYUV                            = 0x04000000,
    /**
     * 8 bit Packed A8B8G8R8. This is a word-ordered format
     * where a pixel is represented by a 32-bit word with R
     * in the lowest 8 bits, G in the next 8 bits, B in the
     * 8 bits after that and A in the highest 8 bits.
     */
    ABGR                            = 0x10000000,
    /**
     * 10 bit Packed A2B10G10R10. This is a word-ordered format
     * where a pixel is represented by a 32-bit word with R
     * in the lowest 10 bits, G in the next 10 bits, B in the
     * 10 bits after that and A in the highest 2 bits.
     */
    ABGR10                          = 0x20000000,
}

type ErrBack = (err?: Error, buf?: ArrayBuffer) => void;

function encodeBuffer(this: NvEncoder, callback: ErrBack): void;
function encodeBuffer(this: NvEncoder, source: any, callback?: ErrBack): void;
function encodeBuffer(this: NvEncoder, ...args: any[]) {
    return (<any> NvEncoder.prototype).encodeBuffer.call(this, ...args);
}

export class CUDAEncoder extends NvEncoder {
    constructor(options: NVEncoderOptions) {
        super({ ...options, deviceType: NvEncoderDeviceType.CUDA });
    }
    encodeFrame(options: { source: any; } | null | undefined, callback: ErrBack): void {
        !options ?
            encodeBuffer.call(this, callback) :
            encodeBuffer.call(this, options.source, callback);
    }
}

function encodeTexture(this: NvEncoder, callback: ErrBack): void;
function encodeTexture(this: NvEncoder, texture: any, target?: number, callback?: ErrBack): void;
function encodeTexture(this: NvEncoder, ...args: any[]) {
    return (<any> NvEncoder.prototype).encodeTexture.call(this, ...args);
}

export class OpenGLEncoder extends NvEncoder {
    constructor(opts: NVEncoderOptions) {
        super({ ...opts, deviceType: NvEncoderDeviceType.OPENGL });
    }
    encodeFrame(options: { texture: any; target: number; } | null | undefined, callback: ErrBack): void {
        !options ?
            encodeTexture.call(this, callback) :
            encodeTexture.call(this, options.texture, options.target, callback);
    }
}
