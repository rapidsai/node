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

import NVENCODER from '../addon';
import { CUDA } from '@nvidia/cuda';
import { NvEncoderOptions, NvEncoderBufferFormat } from '../interfaces';
import { Transform as TransformStream, TransformOptions } from 'stream';

interface CUDANvEncoderConstructor {
    readonly prototype: CUDANvEncoder;
    new(options: NvEncoderOptions): CUDANvEncoder;
}

interface CUDANvEncoder {
    readonly constructor: CUDANvEncoderConstructor;
    readonly frameSize: number;
    readonly bufferCount: number;
    readonly bufferFormat: NvEncoderBufferFormat;
    close(cb: ErrBack): void;
    encode(cb: ErrBack): void;
    copyFromArray(array: any, format?: NvEncoderBufferFormat): void;
    copyFromHostBuffer(buffer: any, format?: NvEncoderBufferFormat): void;
    copyFromDeviceBuffer(buffer: any, format?: NvEncoderBufferFormat): void;
}

export class CUDAEncoder extends (<CUDANvEncoderConstructor> NVENCODER.CUDANvEncoder) {
    constructor(options: NvEncoderOptions) {
        super({ format: NvEncoderBufferFormat.ABGR, ...options });
    }
}

type ErrBack = (err?: Error, buf?: ArrayBuffer) => void;

export class CUDAEncoderTransform extends TransformStream {
    private _encoder: CUDAEncoder;
    constructor({ width, height, format = NvEncoderBufferFormat.ABGR, ...opts }: NvEncoderOptions & TransformOptions) {
        const encoder = new CUDAEncoder({ width, height, format });
        super({
            writableObjectMode: true,
            writableHighWaterMark: 1,
            readableObjectMode: false,
            readableHighWaterMark: encoder.frameSize,
            ...opts,
        });
        this._encoder = encoder;
    }
    get frameSize() { return this._encoder.frameSize; }
    get bufferCount() { return this._encoder.bufferCount; }
    get bufferFormat() { return this._encoder.bufferFormat; }
    // TODO
    resize(_size: { width: number, height: number }) {}
    _transform(source: any, _encoding: string, cb: ErrBack) {
        if (!source) return cb();
        if (!this._copyToFrame(source)) return cb();
        this._encoder.encode((err, ...buffers) => {
            if (err) return cb(err);
            buffers.forEach((b) => this.push(b));
            cb();
        });
    }
    _final(cb: ErrBack) {
        this._encoder.close((err, ...buffers) => {
            if (err) return cb(err);
            buffers.forEach((b) => this.push(b));
            cb();
        });
    }
    _copyToFrame({ texture, buffer, format }: any) {
        if (texture && texture.handle) {
            this._copyTextureToFrame(texture, format);
            return true;
        } else if (buffer) {
            if (buffer._ || buffer.handle) {
                this._copyGLBufferToFrame(buffer.handle ? buffer : { handle: buffer }, format);
                return true;
            } else if (buffer.ptr) {
                this._copyDeviceBufferToFrame(buffer, format);
                return true;
            } else if (ArrayBuffer.isView(buffer) || buffer instanceof ArrayBuffer) {
                this._copyHostBufferToFrame(buffer, format);
                return true;
            }
        }
        return false;
    }
    _copyTextureToFrame(texture: any, format = NvEncoderBufferFormat.ABGR) {
        const resource = getRegisteredTextureResource(texture);
        CUDA.gl.mapResources([resource]);
        const src = CUDA.gl.getMappedArray(resource);
        this._encoder.copyFromArray(src.ary, format);
        CUDA.gl.unmapResources([resource]);
    }
    _copyGLBufferToFrame(buffer: any, format = NvEncoderBufferFormat.ABGR) {
        const resource = getRegisteredBufferResource(buffer);
        CUDA.gl.mapResources([resource]);
        this._encoder.copyFromDeviceBuffer(CUDA.gl.getMappedPointer(resource), format);
        CUDA.gl.unmapResources([resource]);
    }
    _copyHostBufferToFrame(buffer: any, format = NvEncoderBufferFormat.ABGR) {
        this._encoder.copyFromHostBuffer(buffer, format);
    }
    _copyDeviceBufferToFrame(buffer: any, format = NvEncoderBufferFormat.ABGR) {
        this._encoder.copyFromDeviceBuffer(buffer, format);
    }
}

function getRegisteredBufferResource({ handle }: any) {
    if (handle && handle.cudaGraphicsResource === undefined) {
        handle.cudaGraphicsResource = CUDA.gl.registerBuffer(handle.ptr, 0);
    }
    return handle.cudaGraphicsResource;
}

function getRegisteredTextureResource({ handle, target }: any) {
    if (handle && handle.cudaGraphicsResource === undefined) {
        handle.cudaGraphicsResource = CUDA.gl.registerImage(handle.ptr, target, 0);
    }
    return handle.cudaGraphicsResource;
}
