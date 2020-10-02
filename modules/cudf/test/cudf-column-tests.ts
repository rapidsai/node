import { Column } from '@nvidia/cudf';
import { DeviceBuffer } from '@nvidia/rmm';


test('Column initialization', () => {
    const buffer_size = 100;
    const db = new DeviceBuffer(buffer_size,10);
    const col = new Column('int32', 10, db);

    expect(col.type()).toBe('int32');
    expect(col.size()).toBe(buffer_size);
    expect(col.null_count()).toBe(0);
    expect(col.has_nulls()).toBe(false);
    expect(col.nullable()).toBe(false);
});
