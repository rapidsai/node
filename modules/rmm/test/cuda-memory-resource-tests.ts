import { CudaMemoryResource } from '@nvidia/rmm';

test(`CudaMemoryResource initialization`, () => {
    const mr = new CudaMemoryResource();
    expect(mr.supports_streams).toEqual(false);
    expect(mr.supports_get_mem_info).toEqual(true);
});

test(`CudaMemoryResource get_mem_info`, () => {
    const mr = new CudaMemoryResource();
    const [free, avail] = mr.get_mem_info(0);
    expect(typeof free).toBe('number');
    expect(typeof avail).toBe('number');
});

test(`CudaMemoryResource allocate and deallocate`, () => {
    const mr = new CudaMemoryResource();
    const start = mr.get_mem_info(0);
    const ptr = mr.allocate(1000000);
    const end = mr.get_mem_info(0);
    expect(start[0] - end[0]).toBeGreaterThanOrEqual(1000000);
    mr.deallocate(ptr, 1000000);
    const final = mr.get_mem_info(0);
    expect(final[0]).toEqual(start[0]);
});

test(`CudaMemoryResource is_equal`, () => {
    const mr1 = new CudaMemoryResource();
    const mr2 = new CudaMemoryResource();
    expect(mr1.is_equal(mr1)).toEqual(true);
    expect(mr1.is_equal(mr2)).toEqual(true);
    expect(mr2.is_equal(mr1)).toEqual(true);
    expect(mr2.is_equal(mr2)).toEqual(true);
});
