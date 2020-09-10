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

import { CUDA } from '@nvidia/cuda';
import { Transform as TransformStream, TransformOptions } from 'stream';
import { CUDANvEncoder, NvEncoderOptions, NvEncoderBufferFormat } from './nvencoder';

export class CUDAEncoder extends CUDANvEncoder {
    constructor(options: NvEncoderOptions) {
        super({ format: NvEncoderBufferFormat.ABGR, ...options });
    }
}

type ErrBack = (err?: Error, buf?: ArrayBuffer) => void;

export class CUDAEncoderTransform extends TransformStream {
    private _encoder: CUDAEncoder;
    constructor({ width, height, format = NvEncoderBufferFormat.ABGR, ...opts }: NvEncoderOptions & TransformOptions) {
        super({
            ...opts,
            writableObjectMode: true,
            readableObjectMode: false,
        });
        this._encoder = new CUDAEncoder({ width, height, format });
    }
    get encoderBufferCount() { return this._encoder.encoderBufferCount; }
    // TODO
    resize(_size: { width: number, height: number }) {}
    _copyToFrame(source: any) {
        if (!source) { return; }
        else if (source.texture) {
            const resource = getRegisteredTextureResource(source.texture);
            CUDA.gl.mapResources([resource]);
            const src = CUDA.gl.getMappedArray(resource);
            this._encoder.copyFromArray(src.ary);
            CUDA.gl.unmapResources([resource]);
        } else if (source.handle) {
            const resource = getRegisteredBufferResource(source);
            CUDA.gl.mapResources([resource]);
            this._encoder.copyFromDeviceBuffer(CUDA.gl.getMappedPointer(resource));
            CUDA.gl.unmapResources([resource]);
        } else if (ArrayBuffer.isView(source) || source instanceof ArrayBuffer) {
            this._encoder.copyFromHostBuffer(source);
        } else if (source.buffer || source.ptr) {
            this._encoder.copyFromDeviceBuffer(source);
        }
    }
    _transform(source: any, _encoding: string, cb: ErrBack) {
        if (!source) return cb();
        this._copyToFrame(source);
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
}

function getRegisteredBufferResource({ handle }: any) {
    if (handle && handle.cudaGraphicsResource === undefined) {
        handle.cudaGraphicsResource = CUDA.gl.registerBuffer(handle._, 0);
    }
    return handle.cudaGraphicsResource;
}

function getRegisteredTextureResource({ handle, target }: any) {
    if (handle && handle.cudaGraphicsResource === undefined) {
        handle.cudaGraphicsResource = CUDA.gl.registerImage(handle._, target, 0);
    }
    return handle.cudaGraphicsResource;
}
