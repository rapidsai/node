import { CUDAMemory } from '@nvidia/cuda';

describe('CUDAMemory.allocHost', () => {
    test('CUDAMemory.allocHost allocates pinned host memory', () => {
        const hmem = CUDAMemory.allocHost(16);
        expect(hmem.buffer).toBeInstanceOf(ArrayBuffer);
        expect(hmem.byteOffset).toBe(0);
        expect(hmem.byteLength).toBe(16);
    });
});
