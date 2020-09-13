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
import { NvEncoderOptions, NvEncoderBufferFormat } from '../interfaces';
import { Transform as TransformStream, TransformOptions } from 'stream';

type ErrBack = (err?: Error) => void;

interface GLNvEncoderConstructor {
    readonly prototype: GLNvEncoder;
    new(options: NvEncoderOptions): GLNvEncoder;
}

interface GLNvEncoder {
    readonly constructor: GLNvEncoderConstructor;
    readonly frameSize: number;
    readonly bufferCount: number;
    readonly bufferFormat: NvEncoderBufferFormat;
    close(cb: ErrBack): void;
    encode(cb: ErrBack): void;
    texture(): TextureInputFrame;
}

export interface TextureInputFrame {
    readonly pitch: number;
    readonly format: NvEncoderBufferFormat;
    readonly target: number;
    readonly texture: number;
}

export class GLEncoder extends (<GLNvEncoderConstructor> NVENCODER.GLNvEncoder) {
    constructor(options: NvEncoderOptions) {
        super({ format: NvEncoderBufferFormat.ABGR, ...options });
    }
}

export class GLEncoderTransform extends TransformStream {
    private _encoder: GLEncoder;
    constructor({ width, height, format = NvEncoderBufferFormat.ABGR, ...opts }: NvEncoderOptions & TransformOptions) {
        const encoder = new GLEncoder({ width, height, format });
        super({
            writableHighWaterMark: 1,
            writableObjectMode: true,
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
    _copyToFrame(source: any) {
        const target = this._encoder.texture();
        const gl = source.gl as WebGL2RenderingContext;
        gl.bindTexture(target.target, target.texture);
        gl.bindFramebuffer(gl.FRAMEBUFFER, source.handle);
        gl.copyTexImage2D(target.target, 0, source.dataFormat, 0, 0, source.width, source.height, 0);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.bindTexture(target.target, null);
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
