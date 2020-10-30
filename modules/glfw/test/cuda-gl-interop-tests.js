test('CUDA-GL interop', () => {

    require('@nvidia/glfw').createWindow('(' + testCUDAGLInterop.toString() + ')()', true).open();

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

        const cudaBuf = Uint8Buffer.alloc(16).copyFrom(hostBuf).copyInto(hostResult1);

        const lumaBuf = new GLBuffer(gl, { target: gl.ARRAY_BUFFER, accessor: {size: 1, type: gl.UNSIGNED_BYTE}});

        lumaBuf.reallocate(cudaBuf.length * lumaBuf.accessor.BYTES_PER_VERTEX);

        const cudaGLPtr = CUDA.gl.registerBuffer(lumaBuf.handle.ptr, 0);
        CUDA.gl.mapResources([cudaGLPtr]);

        new Uint8Buffer(CUDA.gl.getMappedPointer(cudaGLPtr))
            .copyFrom(hostBuf).copyInto(hostResult2);

        CUDA.gl.unmapResources([cudaGLPtr]);
        CUDA.gl.unregisterResource(cudaGLPtr);

        lumaBuf.getData({dstData: hostResult3});

        assert.ok(hostResult1.equals(hostBuf));
        assert.ok(hostResult2.equals(hostBuf));
        assert.ok(hostResult3.equals(hostBuf));
    }
});
