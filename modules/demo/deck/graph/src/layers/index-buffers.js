import { Buffer, Transform } from '@luma.gl/core';

const computeIndexBufferVS = `\
#version 300 es
flat out uint outIndex;
void main() {
    outIndex = uint(gl_VertexID);
}
`;

export function createIndexBufferTransform(gl) {
    return new Transform(gl, {
        isInstanced: false,
        varyings: ['outIndex'],
        vs: computeIndexBufferVS,
        feedbackBuffers: { outIndex: new Buffer(gl, 1) },
    });
}

const bufferCache = new Map();

export function computeIndexBuffers(gl, count, computeIndexBuffer) {
    if (bufferCache.has(count)) {
        return bufferCache.get(count);
    }
    const accessor = { size: 1, type: gl.INT };
    const byteLength = count * Uint32Array.BYTES_PER_ELEMENT;

    const outIndex = new Buffer(gl, { byteLength, accessor });
    computeIndexBuffer.update({ elementCount: count, feedbackBuffers: { outIndex } });
    computeIndexBuffer.run({ offset: 0 });

    const indices = outIndex.getData();
    const opts = { byteLength, accessor, target: gl.ELEMENT_ARRAY_BUFFER, };
    const normalIndex = new Buffer(gl, { ...opts, data: indices });
    const randomIndex = new Buffer(gl, { ...opts, data: shuffle(indices) });

    return { normalIndex, randomIndex };
}

const shuffle = (arr) => {
    const { random, floor } = Math;
    for(let i = arr.length; --i > -1;) {
        const j = floor(random() * i);
        const val = arr[i];
        arr[i] = arr[j];
        arr[j] = val;
    }
    return arr;
}
