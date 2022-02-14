// Copyright (c) 2022, NVIDIA CORPORATION.
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

const {CUDA, Uint8Buffer}               = require('@rapidsai/cuda');
const {Buffer: DeckBuffer}              = require('@rapidsai/deck.gl');
const {Framebuffer, readPixelsToBuffer} = require('@luma.gl/webgl');
const shm                               = require('shm-typed-array');

Object.defineProperty(Framebuffer, Symbol.hasInstance, {
  value: (x) => x?.constructor?.name === 'Framebuffer',
});

module.exports = copyAndConvertFramebuffer;

function copyAndConvertFramebuffer() {
  let i420DeviceBuffer = null;
  let rgbaDeviceBuffer = null;

  return ({gl, framebuffer}, sharedMemoryKey) => {
    const {width, height} = framebuffer;
    const rgbaByteLength  = width * height * 4;
    const i420ByteLength  = width * height * 3 / 2;

    if (rgbaDeviceBuffer?.byteLength !== rgbaByteLength) {
      rgbaDeviceBuffer?.delete({deleteChildren: true});
      rgbaDeviceBuffer = new DeckBuffer(gl, {
        byteLength: rgbaByteLength,
        accessor: {type: gl.UNSIGNED_BYTE, size: 4},
      });
    }

    if (i420DeviceBuffer?.byteLength !== i420ByteLength) {
      i420DeviceBuffer = new Uint8Buffer(i420ByteLength);
    }

    // DtoD copy from framebuffer into our pixelbuffer
    readPixelsToBuffer(
      framebuffer, {sourceType: gl.UNSIGNED_BYTE, sourceFormat: gl.RGBA, target: rgbaDeviceBuffer});

    // Map and unmap the GL buffer as a CUDA buffer
    rgbaDeviceBuffer.asMappedResource((glBuffer) => {
      const cuBuffer = glBuffer.asCUDABuffer(0, rgbaByteLength);
      // flip horizontally to account for WebGL's coordinate system (e.g. ffmpeg -vf vflip)
      CUDA.rgbaMirror(width, height, 0, cuBuffer);
      // convert colorspace from OpenGL's BGRA to WebRTC's IYUV420
      CUDA.bgraToYCrCb420(i420DeviceBuffer, cuBuffer, width, height);
    });

    // DtoH copy for output
    const out = shm.get(sharedMemoryKey, 'Uint8ClampedArray');
    i420DeviceBuffer.copyInto(out);
    shm.detach(sharedMemoryKey);

    return {width, height, data: sharedMemoryKey};
  };
}
