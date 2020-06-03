import { CUDAMemory } from '@nvidia/cuda';

describe('CUDAMemory.as', () => {
    test('CUDAMemory.as wraps a host buffer in a CUDAMemory', () => {
        const hmem = CUDAMemory.as(Buffer.alloc(16).fill(7));
        expect(hmem.buffer).toBeInstanceOf(ArrayBuffer);
        expect(hmem.byteOffset).toBe(0);
        expect(hmem.byteLength).toBe(16);
    });
});
