import { Table, Column, TypeId } from '@nvidia/cudf';
import { DeviceBuffer, CudaMemoryResource } from '@nvidia/rmm';
import { Uint8Buffer, Int32Buffer, setDefaultAllocator } from '@nvidia/cuda';

const mr = new CudaMemoryResource();

setDefaultAllocator((byteLength) => new DeviceBuffer(byteLength, 0, mr));

test('Column initialization', () => {
    const length = 100;
    const col_0 = new Column({ type: TypeId.INT32, data: new Int32Buffer(length) });

    const col_1 = new Column({
        type: TypeId.BOOL8,
        data: new Uint8Buffer(length),
        nullMask: new Uint8Buffer(64),
    });
    const table_0 = new Table({"col_0":col_0, "col_1": col_1});
    expect(table_0.numColumns).toBe(2);
    expect(table_0.numRows).toBe(length);
    expect(table_0.columns).toStrictEqual([ 'col_0', 'col_1' ]);
    expect(table_0["col_0"].type.id).toBe(col_0.type.id);
    expect(table_0["col_1"].type.id).toBe(col_1.type.id);
});
