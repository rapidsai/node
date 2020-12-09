import { expect, test } from '@jest/globals';

import { Uint8Buffer, devices } from '@nvidia/cuda';

import {
    DeviceBuffer,
    CudaMemoryResource,
    ManagedMemoryResource,
    PoolMemoryResource,
    FixedSizeMemoryResource,
    BinningMemoryResource,
    LoggingResourceAdapter,
} from '@nvidia/rmm';

const _1MiB = 1 << 20;
const _2MiB = 1 << 21;
const _4MiB = 1 << 23;
const _16MiB = 1 << 24;

const memoryResourceTestConfigs = [
    { 
        name: 'CudaMemoryResource (no device)',
        supportsStreams: false, supportsGetMemInfo: true, comparable: true,
        createMemoryResource: () => new CudaMemoryResource(),
    },
    ...[...devices].map((_, i) => ({
        name: `CudaMemoryResource (device ${i})`,
        supportsStreams: false, supportsGetMemInfo: true, comparable: true,
        createMemoryResource: () => new CudaMemoryResource(i),
    })),
    {
        name: 'ManagedMemoryResource',
        supportsStreams: false, supportsGetMemInfo: true, comparable: true,
        createMemoryResource: () => new ManagedMemoryResource(),
    },
    {
        name: 'PoolMemoryResource',
        supportsStreams: true, supportsGetMemInfo: false, comparable: false,
        createMemoryResource: () => new PoolMemoryResource(new CudaMemoryResource(), _1MiB, _16MiB),
    },
    {
        name: 'FixedSizeMemoryResource',
        supportsStreams: true, supportsGetMemInfo: false, comparable: false,
        createMemoryResource: () => new FixedSizeMemoryResource(new CudaMemoryResource(), _4MiB, 1),
    },
    {
        name: 'BinningMemoryResource',
        supportsStreams: true, supportsGetMemInfo: false, comparable: false,
        createMemoryResource: () => new BinningMemoryResource(new CudaMemoryResource(), 20, 23),
    },
    {
        name: 'LoggingResourceAdapter',
        supportsStreams: false, supportsGetMemInfo: true, comparable: true,
        createMemoryResource: () => new LoggingResourceAdapter(new CudaMemoryResource(), '/dev/stdout', true),
    },
];

memoryResourceTestConfigs.forEach(({ name, supportsStreams, supportsGetMemInfo, comparable, createMemoryResource }) => {
    describe(name, () => {
        test(`construction`, () => {
            const mr = createMemoryResource();
            expect(mr.supportsStreams).toEqual(supportsStreams);
            expect(mr.supportsGetMemInfo).toEqual(supportsGetMemInfo);
        });

        test(`getMemInfo()`, () => {
            const mr = createMemoryResource();
            const memoryInfo = mr.getMemInfo(0);
            expect(Array.isArray(memoryInfo)).toBe(true);
            expect(memoryInfo.length).toBe(2);
            memoryInfo.forEach((v) => expect(typeof v).toBe('number'));
        });

        test(`isEqual()`, () => {
            const mr1 = createMemoryResource();
            const mr2 = createMemoryResource();
            expect(mr1.isEqual(mr1)).toEqual(true);
            expect(mr1.isEqual(mr2)).toEqual(comparable);
            expect(mr2.isEqual(mr1)).toEqual(comparable);
            expect(mr2.isEqual(mr2)).toEqual(true);
        });

        test(`works with DeviceBuffer`, () => {
            const mr = createMemoryResource();
            let free0 = 0, free1 = 0;
            let total0 = 0, total1 = 0;
            mr.supportsGetMemInfo && ([free0, total0] = mr.getMemInfo(0));
            // Fill the buffer with 1s, because CUDA Managed
            // memory is only allocated when it's actually used.
            // @ts-ignore
            let buf = new Uint8Buffer(new DeviceBuffer(_2MiB, 0, mr)).fill(1);
            mr.supportsGetMemInfo && ([free1, total1] = mr.getMemInfo(0));
            expect(total0).toEqual(total1);
            if (mr.supportsGetMemInfo) {
                expect(free0 - free1).toBeGreaterThanOrEqual(_2MiB);
            }
            buf = null!;
        });
    });
});
