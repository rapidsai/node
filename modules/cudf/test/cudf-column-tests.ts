import { Column, types } from '@nvidia/cudf';
import { DeviceBuffer } from '@nvidia/rmm';


test('Column initialization', () => {
    const buffer_size = 100;
    const db = new DeviceBuffer(buffer_size,10);
    const col = new Column(types.INT32, 10, db);

    expect(col.type()).toBe(types.INT32);
    expect(col.size()).toBe(buffer_size);
    expect(col.nullCount()).toBe(0);
    expect(col.hasNulls()).toBe(false);
    expect(col.nullable()).toBe(false);
});


test('Column initialization with null_mask', () => {
    const buffer_size = 100;
    const db = new DeviceBuffer(buffer_size,10);
    const null_mask = new DeviceBuffer(buffer_size,10);
    const col = new Column(types.BOOL8, 10, db, null_mask);

    expect(col.type()).toBe(types.BOOL8);
    expect(col.size()).toBe(buffer_size);
    expect(col.nullCount()).toBe(100);
    expect(col.hasNulls()).toBe(true);
    expect(col.nullable()).toBe(true);
});


test('Column null_mask, null_count', () => {
    const buffer_size = 100;
    const db = new DeviceBuffer(buffer_size,10);
    const null_mask = new DeviceBuffer(buffer_size,10);
    const col = new Column(types.FLOAT32, 10, db, null_mask, 1);

    expect(col.type()).toBe(types.FLOAT32);
    expect(col.size()).toBe(buffer_size);
    expect(col.nullCount()).toBe(1);
    expect(col.hasNulls()).toBe(true);
    expect(col.nullable()).toBe(true);
});


test('test child(child_index), num_children', () => {
    const buffer_size = 100;
    const db = new DeviceBuffer(buffer_size,10);
    const db1 = new DeviceBuffer(buffer_size*2,10);
    const null_mask = new DeviceBuffer(buffer_size,10);
    const col = new Column(types.FLOAT32, 10, db, null_mask, 1);

    const col1 = new Column(types.FLOAT64, 10, db1, null_mask, 20, [col]);

    expect(col1.type()).toBe(types.FLOAT64);
    expect(col1.numChildren()).toBe(1);
    expect(col1.child(0).size()).toBe(col.size());
    expect(col1.child(0).type()).toBe(col.type());
});


test('test Column(column) constructor', () => {
    const buffer_size = 100;
    const db = new DeviceBuffer(buffer_size,10);

    const null_mask = new DeviceBuffer(buffer_size,10);
    const col = new Column(types.FLOAT32, 10, db, null_mask, 1);
    const col1 = new Column(col);

    expect(col1.type()).toBe(types.FLOAT32);
    expect(col1.size()).toBe(buffer_size);
    expect(col1.nullCount()).toBe(1);
    expect(col1.hasNulls()).toBe(true);
    expect(col1.nullable()).toBe(true);
});
