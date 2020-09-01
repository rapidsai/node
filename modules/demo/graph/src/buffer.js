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

import { Buffer as DeckBuffer } from '@luma.gl/webgl';

export const Buffer = ((Buffer) => {
    if (process.env.REACT_APP_ENVIRONMENT === 'browser') {
        return class DeckBuffer extends Buffer {
            static mapResources(buffers = []) {}
            static unmapResources(buffers = []) {}
        };
    } else {
        const { CUDA, CUDAUint8Array } = require('@nvidia/cuda');
        return class CUDABuffer extends Buffer {
            static mapResources(buffers = []) {
                buffers = buffers.filter((buffer) =>
                    buffer && buffer.handle &&
                    buffer.handle.cudaGraphicsResource !== undefined &&
                    buffer.handle.cudaGraphicsResourceMapped === false);
                CUDA.gl.mapResources(buffers.map((buffer) => buffer.handle.cudaGraphicsResource));
                buffers.forEach((buffer) => buffer.handle.cudaGraphicsResourceMapped = true);
            }
            static unmapResources(buffers = []) {
                buffers = buffers.filter((buffer) =>
                    buffer && buffer.handle &&
                    buffer.handle.cudaGraphicsResource !== undefined &&
                    buffer.handle.cudaGraphicsResourceMapped === true);
                CUDA.gl.unmapResources(buffers.map((buffer) => buffer.handle.cudaGraphicsResource));
                buffers.forEach((buffer) => buffer.handle.cudaGraphicsResourceMapped = false);
            }
            constructor(...args) {
                super(...args);
                this._registerResource(this.handle);
            }
            subData(props = {}) {
                if (!this._handle || !this._handle.cudaGraphicsResourceMapped) {
                    return super.subData(props);
                }
                props = props instanceof ArrayBuffer ? { data: new Uint8Array(props), length: props.byteLength }
                         : ArrayBuffer.isView(props) ? { data: props, length: props.byteLength }
                                                     : { data: new Uint8Array(0), ...props };
                const {
                    data, offset = 0, srcOffset = 0,
                    length = data.byteLength,
                    byteLength = length,
                } = props;
                const ptr = CUDA.gl.getMappedPointer(this._handle.cudaGraphicsResource);
                const ary = new CUDAUint8Array(ptr, offset, this.byteLength - offset);
                ary.set({ buffer: data, byteOffset: srcOffset, byteLength });
                return this;
            }
            _deleteHandle(handle = this._handle) {
                return this._unregisterResource(handle)._deleteHandle(handle);
            }
            _setData(data, offset = 0, byteLength = data.byteLength + offset) {
                const mapped = this._handle.cudaGraphicsResourceMapped;
                mapped && this._unmapResource(this._handle);
                super._setData(data, offset, byteLength);
                mapped && this._mapResource(this._handle);
                return this;
            }
            _setByteLength(byteLength, usage = this.usage) {
                const mapped = this._handle.cudaGraphicsResourceMapped;
                mapped && this._unmapResource(this._handle);
                super._setByteLength(byteLength, usage);
                mapped && this._mapResource(this._handle);
                return this;
            }
            _registerResource(handle = this._handle) {
                if (handle && handle.cudaGraphicsResource === undefined) {
                    this._handle = handle;
                    handle.cudaGraphicsResourceMapped = false;
                    handle.cudaGraphicsResource = CUDA.gl.registerBuffer(handle._, 0);
                }
                return this;
            }
            _unregisterResource(handle = this._handle) {
                if (handle && handle.cudaGraphicsResource !== undefined) {
                    this._unmapResource(handle);
                    CUDA.gl.unregisterResource(handle.cudaGraphicsResource);
                    handle.cudaGraphicsResource = undefined;
                }
                return this;
            }
            _mapResource(handle = this._handle) {
                if (handle && !handle.cudaGraphicsResourceMapped) {
                    CUDA.gl.mapResources([handle.cudaGraphicsResource]);
                    handle.cudaGraphicsResourceMapped = true;
                }
                return this;
            }
            _unmapResource(handle = this._handle) {
                if (handle && handle.cudaGraphicsResourceMapped) {
                    CUDA.gl.unmapResources([handle.cudaGraphicsResource]);
                    handle.cudaGraphicsResourceMapped = false;
                }
                return this;
            }
        };
    }
})(DeckBuffer);
