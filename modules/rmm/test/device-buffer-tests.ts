import { DeviceBuffer } from '@nvidia/rmm';

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
