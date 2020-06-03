import { CUDAMemory } from '@nvidia/cuda';

describe('CUDAMemory.prototype.fill', () => {
    test('fills device memory with a value', () => {
        const hmem = Buffer.alloc(16);
        CUDAMemory.alloc(16).fill(7).copyInto(hmem);
        expect(hmem).toEqual(Buffer.alloc(16).fill(7));
    });
});

describe('CUDAMemory.prototype.fillAsync', () => {
    test('fills device memory with a value', async () => {
        const hmem = Buffer.alloc(16);
        (await CUDAMemory.alloc(16).fillAsync(7)).copyInto(hmem);
        expect(hmem).toEqual(Buffer.alloc(16).fill(7));
    });
});
