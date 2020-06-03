import { CUDAMemory } from '@nvidia/cuda';

describe('CUDAMemory.prototype.copyFrom', () => {
    test('copies between unregistered host memory and device memory', () => {
        const hmem = Buffer.alloc(16).fill(7);
        const data = Buffer.alloc(16);
        const dmem = CUDAMemory.alloc(16);
        dmem.copyFrom(hmem).copyInto(data);
        expect(data).toEqual(Buffer.alloc(16).fill(7));
    });
});

describe('CUDAMemory.prototype.copyFromAsync', () => {
    test('copies between unregistered host memory and device memory', async () => {
        const hmem = Buffer.alloc(16).fill(7);
        const data = Buffer.alloc(16);
        const dmem = CUDAMemory.alloc(16);
        await dmem.copyFromAsync(hmem);
        dmem.copyInto(data);
        expect(data).toEqual(Buffer.alloc(16).fill(7));
    });
});
