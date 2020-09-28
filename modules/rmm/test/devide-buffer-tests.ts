import { DeviceBuffer } from '@nvidia/rmm'

import { devices, CUDADevice, CUDADeviceFlag } from '@nvidia/cuda';

test(`DeviceBuffer initialization`, () => {
    const db = new DeviceBuffer();
    expect(db.byteLength).toBe(0);
    expect(db.capacity).toBe(0);
    expect(db.isEmpty).toBe(true);
});

test(`DeviceBuffer resize`, () => {
    const db = new DeviceBuffer();
    db.resize(1234);
    expect(db.byteLength).toBe(1234);
    expect(db.capacity).toBe(1234);
    expect(db.isEmpty).toBe(false);
});
