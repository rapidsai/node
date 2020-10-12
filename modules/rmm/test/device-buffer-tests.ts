import { DeviceBuffer } from '@nvidia/rmm';

test(`DeviceBuffer empty initialization`, () => {
    const db = new DeviceBuffer(0);
    expect(db.byteLength).toBe(0);
    expect(db.capacity).toBe(0);
    expect(db.isEmpty).toBe(true);
});

test(`DeviceBuffer initialization`, () => {
    const db = new DeviceBuffer(1000);
    expect(db.byteLength).toBe(1000);
    expect(db.capacity).toBe(1000);
    expect(db.isEmpty).toBe(false);
});

test(`DeviceBuffer resize`, () => {
    const db = new DeviceBuffer(1000);
    db.resize(1234);
    expect(db.byteLength).toBe(1234);
    expect(db.capacity).toBe(1234);
    expect(db.isEmpty).toBe(false);

    db.resize(0);
    expect(db.byteLength).toBe(0);
    expect(db.capacity).toBe(1234);
    expect(db.isEmpty).toBe(true);
});
