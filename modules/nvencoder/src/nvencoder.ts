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

type ErrBack = (err?: Error, buf?: ArrayBuffer) => void;

export interface NvEncoderOptions {
    width: number;
    height: number;
    format?: NvEncoderBufferFormat;
    // hevc: boolean;
    // device: number;
    // averageBitRate?: number;
    // maxBitRate?: number;
    // maxWidth?: number;
    // maxHeight?: number;
    // frameRateNumerator?: number;
    // frameRateDenomenator?: number;
}

export interface GLNvEncoderConstructor {
    readonly prototype: GLNvEncoder;
    new(options: NvEncoderOptions): GLNvEncoder;
}

export interface GLNvEncoder {
    readonly constructor: GLNvEncoderConstructor;
    readonly encoderBufferCount: number;
    close(cb: ErrBack): void;
    encode(cb: ErrBack): void;
    texture(): TextureInputFrame;
}

export const GLNvEncoder: GLNvEncoderConstructor = NVENCODER.GLNvEncoder;

export interface CUDANvEncoderConstructor {
    readonly prototype: CUDANvEncoder;
    new(options: NvEncoderOptions): CUDANvEncoder;
}

export interface CUDANvEncoder {
    readonly constructor: CUDANvEncoderConstructor;
    readonly encoderBufferCount: number;
    close(cb: ErrBack): void;
    encode(cb: ErrBack): void;
    copyFromArray(array: any): void;
    copyFromHostBuffer(buffer: any): void;
    copyFromDeviceBuffer(buffer: any): void;
}

export const CUDANvEncoder: CUDANvEncoderConstructor = NVENCODER.CUDANvEncoder;

interface InputFrame {
    readonly pitch: number;
    readonly format: NvEncoderBufferFormat;
}

export interface ArrayInputFrame extends InputFrame {
    readonly array: number;
}

export interface BufferInputFrame extends InputFrame {
    readonly buffer: number;
    readonly byteLength: number;
}

export interface TextureInputFrame extends InputFrame {
    readonly target: number;
    readonly texture: number;
}

export enum NvEncoderBufferFormat {
    /** Undefined buffer format */
    UNDEFINED                       = NVENCODER.bufferFormats.UNDEFINED,
    /** Semi-Planar YUV [Y plane followed by interleaved UV plane] */
    NV12                            = NVENCODER.bufferFormats.NV12,
    /** Planar YUV [Y plane followed by V and U planes] */
    YV12                            = NVENCODER.bufferFormats.YV12,
    /** Planar YUV [Y plane followed by U and V planes] */
    IYUV                            = NVENCODER.bufferFormats.IYUV,
    /** Planar YUV [Y plane followed by U and V planes] */
    YUV444                          = NVENCODER.bufferFormats.YUV444,
    /**
     * 10 bit Semi-Planar YUV [Y plane followed by interleaved UV plane].
     * Each pixel of size 2 bytes. Most Significant 10 bits contain pixel data.
     */
    YUV420_10BIT                    = NVENCODER.bufferFormats.YUV420_10BIT,
    /**
     * 10 bit Planar YUV444 [Y plane followed by U and V planes].
     * Each pixel of size 2 bytes. Most Significant 10 bits contain pixel data.
     */
    YUV444_10BIT                    = NVENCODER.bufferFormats.YUV444_10BIT,
    /**
     * 8 bit Packed A8R8G8B8. This is a word-ordered format
     * where a pixel is represented by a 32-bit word with B
     * in the lowest 8 bits, G in the next 8 bits, R in the
     * 8 bits after that and A in the highest 8 bits.
     */
    ARGB                            = NVENCODER.bufferFormats.ARGB,
    /**
     * 10 bit Packed A2R10G10B10. This is a word-ordered format
     * where a pixel is represented by a 32-bit word with B
     * in the lowest 10 bits, G in the next 10 bits, R in the
     * 10 bits after that and A in the highest 2 bits.
     */
    ARGB10                          = NVENCODER.bufferFormats.ARGB10,
    /**
     * 8 bit Packed A8Y8U8V8. This is a word-ordered format
     * where a pixel is represented by a 32-bit word with V
     * in the lowest 8 bits, U in the next 8 bits, Y in the
     * 8 bits after that and A in the highest 8 bits.
     */
    AYUV                            = NVENCODER.bufferFormats.AYUV,
    /**
     * 8 bit Packed A8B8G8R8. This is a word-ordered format
     * where a pixel is represented by a 32-bit word with R
     * in the lowest 8 bits, G in the next 8 bits, B in the
     * 8 bits after that and A in the highest 8 bits.
     */
    ABGR                            = NVENCODER.bufferFormats.ABGR,
    /**
     * 10 bit Packed A2B10G10R10. This is a word-ordered format
     * where a pixel is represented by a 32-bit word with R
     * in the lowest 10 bits, G in the next 10 bits, B in the
     * 10 bits after that and A in the highest 2 bits.
     */
    ABGR10                          = NVENCODER.bufferFormats.ABGR10,
}
