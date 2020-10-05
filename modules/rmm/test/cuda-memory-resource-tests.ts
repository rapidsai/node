import { CudaMemoryResource } from '@nvidia/rmm';

test(`CudaMemoryResource initialization`, () => {
    const mr = new CudaMemoryResource();
    expect(mr.supportsStreams).toEqual(false);
    expect(mr.supportsGetMemInfo).toEqual(true);
});

test(`CudaMemoryResource getMemInfo`, () => {
    const mr = new CudaMemoryResource();
    const [free, avail] = mr.getMemInfo(0);
    expect(typeof free).toBe('number');
    expect(typeof avail).toBe('number');
});

test(`CudaMemoryResource allocate and deallocate`, () => {
    const mr = new CudaMemoryResource();
    const start = mr.getMemInfo(0);
    const ptr = mr.allocate(1000000);
    const end = mr.getMemInfo(0);
    expect(start[0] - end[0]).toBeGreaterThanOrEqual(1000000);
    mr.deallocate(ptr, 1000000);
    const final = mr.getMemInfo(0);
    expect(final[0]).toEqual(start[0]);
});

test(`CudaMemoryResource isEqual`, () => {
    const mr1 = new CudaMemoryResource();
    const mr2 = new CudaMemoryResource();
    expect(mr1.isEqual(mr1)).toEqual(true);
    expect(mr1.isEqual(mr2)).toEqual(true);
    expect(mr2.isEqual(mr1)).toEqual(true);
    expect(mr2.isEqual(mr2)).toEqual(true);
});
