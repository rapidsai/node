import { CUDAMemory } from '@nvidia/cuda';

describe('CUDAMemory.alloc', () => {
    test('allocates device memory', () => {
        const dmem = CUDAMemory.alloc(16);
        expect(dmem.buffer).toBeTruthy();
        expect(dmem.byteOffset).toBe(0);
        expect(dmem.byteLength).toBe(16);
    });
});
