import { CUDAMemory } from '@nvidia/cuda';

describe('CUDAMemory.prototype.copyInto', () => {
    test('copies between unregistered host memory and device memory', () => {
        const hmem = Buffer.alloc(16).fill(7);
        const data = Buffer.alloc(16);
        const dmem = CUDAMemory.alloc(16);
        CUDAMemory.as(hmem).copyInto(dmem);
        dmem.copyInto(data);
        expect(data).toEqual(Buffer.alloc(16).fill(7));
    });
});

describe('CUDAMemory.prototype.copyIntoAsync', () => {
    test('copies between unregistered host memory and device memory', async () => {
        const hmem = Buffer.alloc(16).fill(7);
        const data = Buffer.alloc(16);
        const dmem = CUDAMemory.alloc(16);
        await CUDAMemory.as(hmem).copyIntoAsync(dmem);
        dmem.copyInto(data, 0);
        expect(data).toEqual(Buffer.alloc(16).fill(7));
    });
});
