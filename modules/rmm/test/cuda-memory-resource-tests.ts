import { DeviceBuffer, CudaMemoryResource } from '@nvidia/rmm';

test(`CudaMemoryResource initialization`, () => {
    const mr = new CudaMemoryResource();
    expect(mr.supportsStreams).toEqual(false);
    expect(mr.supportsGetMemInfo).toEqual(true);
});

test(`CudaMemoryResource getMemInfo`, () => {
    const mr = new CudaMemoryResource();
    const [free, total] = mr.getMemInfo(0);
    expect(typeof free).toBe('number');
    expect(typeof total).toBe('number');
});

test(`CudaMemoryResource allocate and deallocate`, (done) => {
    const mr = new CudaMemoryResource();
    const [start] = mr.getMemInfo(0);
    let buf = new DeviceBuffer(1000000, 0, mr);
    const [end] = mr.getMemInfo(0);
    expect(start - end).toBeGreaterThanOrEqual(1000000);
    buf = null!;
    setTimeout(() => {
        const [final] = mr.getMemInfo(0);
        expect(final).toEqual(start);
        done();
    }, 1000);
});

test(`CudaMemoryResource isEqual`, () => {
    const mr1 = new CudaMemoryResource();
    const mr2 = new CudaMemoryResource();
    expect(mr1.isEqual(mr1)).toEqual(true);
    expect(mr1.isEqual(mr2)).toEqual(true);
    expect(mr2.isEqual(mr1)).toEqual(true);
    expect(mr2.isEqual(mr2)).toEqual(true);
});
