// Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

import {Buffer as LumaBuffer, BufferProps} from '@luma.gl/webgl';

declare module '@luma.gl/webgl' {
  // Add protected props and methods missing on the luma.gl typings
  class Resource {
    _deleteHandle(handle?: any): void;
  }
  class Buffer {
    // _handle isn't mutable in practice, even though it is in the luma.gl typings
    _handle: any;
    _deleteHandle(): void;
    _setData(data: any, offset?: number, byteLength?: number): void;
    _setByteLength(byteLength: number, usage?: number): void;
  }
}

export const Buffer = (() => {
  if (process.env.REACT_APP_ENVIRONMENT === 'browser') {
    return class DeckBuffer extends LumaBuffer {
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      static mapResources(_buffers: any[] = []) {}
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      static unmapResources(_buffers: any[] = []) {}
    };
  } else {
    const {CUDA, Uint8Buffer} = require('@nvidia/cuda');
    return class CUDABuffer extends LumaBuffer {
      static mapResources(buffers: any[] = []) {
        buffers = buffers.filter((buffer) => buffer && buffer.handle &&
                                             buffer.handle.cudaGraphicsResource !== undefined &&
                                             buffer.handle.cudaGraphicsResourceMapped === false);
        CUDA.runtime.cudaGraphicsMapResources(
          buffers.map((buffer) => buffer.handle.cudaGraphicsResource));
        buffers.forEach((buffer) => buffer.handle.cudaGraphicsResourceMapped = true);
      }
      static unmapResources(buffers: any[] = []) {
        buffers = buffers.filter((buffer) => buffer && buffer.handle &&
                                             buffer.handle.cudaGraphicsResource !== undefined &&
                                             buffer.handle.cudaGraphicsResourceMapped === true);
        CUDA.runtime.cudaGraphicsUnmapResources(
          buffers.map((buffer) => buffer.handle.cudaGraphicsResource));
        buffers.forEach((buffer) => buffer.handle.cudaGraphicsResourceMapped = false);
      }
      constructor(gl: WebGLRenderingContext, props?: BufferProps);
      constructor(gl: WebGLRenderingContext, data: ArrayBufferView|number[]);
      constructor(gl: WebGLRenderingContext, byteLength: number);
      constructor(gl: WebGLRenderingContext, propsDataOrByteLength?: any) {
        super(gl, propsDataOrByteLength);
        if (this.byteLength > 0) { this._registerResource(this.handle); }
      }
      subData(props: any = {}) {
        if (!this._handle || !this._handle.cudaGraphicsResourceMapped) {
          return super.subData(props);
        }
        props = props instanceof ArrayBuffer
                  ? {data: new Uint8Array(props), length: props.byteLength}
                : ArrayBuffer.isView(props) ? {data: props, length: props.byteLength}
                                            : {data: new Uint8Array(0), ...props};
        const {
          data: buffer,
          length                = buffer.byteLength,
          byteLength: srcLength = length,
          offset                = 0,
          srcOffset             = 0,
        } = props;
        this.asCUDABuffer(offset).set({buffer, byteOffset: srcOffset, byteLength: srcLength});
        return this;
      }
      asCUDABuffer(byteOffset = 0, byteLength = this.byteLength - byteOffset) {
        if (this._handle.cudaGraphicsResourceMapped) {
          return new Uint8Buffer(
            CUDA.runtime.cudaGraphicsResourceGetMappedPointer(this._handle.cudaGraphicsResource),
            byteOffset,
            byteLength);
        }
        throw new Error(
          'OpenGL Buffer must be mapped as a CUDAGraphicsResource to create a CUDA buffer');
      }
      _deleteHandle() {
        this._unregisterResource(this._handle);
        return super._deleteHandle();
      }
      // eslint-disable-next-line @typescript-eslint/restrict-plus-operands
      _setData(data: any, offset = 0, byteLength = data.byteLength + offset) {
        const mapped = this._handle.cudaGraphicsResourceMapped;
        mapped && this._unmapResource(this._handle);
        super._setData(data, offset, byteLength);
        mapped && this._mapResource(this._handle);
        return this;
      }
      _setByteLength(byteLength: number, usage = this.usage) {
        const mapped = this._handle.cudaGraphicsResourceMapped;
        mapped && this._unmapResource(this._handle);
        super._setByteLength(byteLength, usage);
        mapped && this._mapResource(this._handle);
        return this;
      }
      _registerResource(handle = this._handle) {
        if (handle && handle.cudaGraphicsResource === undefined) {
          this._handle                      = handle;
          handle.cudaGraphicsResourceMapped = false;
          handle.cudaGraphicsResource = CUDA.runtime.cudaGraphicsGLRegisterBuffer(handle.ptr, 0);
        }
        return this;
      }
      _unregisterResource(handle = this._handle) {
        if (handle && handle.cudaGraphicsResource !== undefined) {
          this._unmapResource(handle);
          CUDA.runtime.cudaGraphicsUnregisterResource(handle.cudaGraphicsResource);
          handle.cudaGraphicsResource = undefined;
        }
        return this;
      }
      _mapResource(handle = this._handle) {
        if (handle && !handle.cudaGraphicsResourceMapped) {
          CUDA.runtime.cudaGraphicsMapResources([handle.cudaGraphicsResource]);
          handle.cudaGraphicsResourceMapped = true;
        }
        return this;
      }
      _unmapResource(handle = this._handle) {
        if (handle && handle.cudaGraphicsResourceMapped) {
          CUDA.runtime.cudaGraphicsUnmapResources([handle.cudaGraphicsResource]);
          handle.cudaGraphicsResourceMapped = false;
        }
        return this;
      }
    };
  }
})();
