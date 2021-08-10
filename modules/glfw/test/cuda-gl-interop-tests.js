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

test.skip('CUDA-GL interop', () => {

  require('@nvidia/glfw')
    .createWindow(testCUDAGLInterop, true)
    .open();

  function testCUDAGLInterop() {

    const assert = require('assert');
    const { Uint8Buffer, CUDA } = require('@nvidia/cuda');
    const { Buffer: GLBuffer } = require('@luma.gl/core');
    const { WebGL2RenderingContext } = require('@nvidia/webgl');

    const gl = new WebGL2RenderingContext();

    const hostResult1 = Buffer.alloc(16);
    const hostResult2 = Buffer.alloc(16);
    const hostResult3 = Buffer.alloc(16);

    const hostBuf = Buffer.alloc(16).fill(7);

    const cudaBuf = new Uint8Buffer(16).copyFrom(hostBuf).copyInto(hostResult1);

    const lumaBuf = new GLBuffer(gl, {
      target: gl.ARRAY_BUFFER,
      accessor: {
        size: 1,
        type: gl.UNSIGNED_BYTE
      }
    });

    lumaBuf.reallocate(cudaBuf.length * lumaBuf.accessor.BYTES_PER_VERTEX);

    const cudaGLPtr = CUDA.runtime.cudaGraphicsGLRegisterBuffer(lumaBuf.handle.ptr, 0);
    CUDA.runtime.cudaGraphicsMapResources([cudaGLPtr]);

    const cudaGLMem = CUDA.runtime.cudaGraphicsResourceGetMappedPointer(cudaGLPtr);

    new Uint8Buffer(cudaGLMem).copyFrom(hostBuf).copyInto(hostResult2);

    CUDA.runtime.cudaGraphicsUnmapResources([cudaGLPtr]);
    CUDA.runtime.cudaGraphicsUnregisterResource(cudaGLPtr);

    lumaBuf.getData({ dstData: hostResult3 });

    assert.ok(hostResult1.equals(hostBuf));
    assert.ok(hostResult2.equals(hostBuf));
    assert.ok(hostResult3.equals(hostBuf));
  }
});
