// Copyright (c) 2020, NVIDIA CORPORATION.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import { Column, TypeId } from '@nvidia/cudf';
import { DeviceBuffer, CudaMemoryResource } from '@nvidia/rmm';
import { Uint8Buffer, Int32Buffer, setDefaultAllocator, Float32Buffer } from '@nvidia/cuda';

const mr = new CudaMemoryResource();

setDefaultAllocator((byteLength) => new DeviceBuffer(byteLength, 0, mr));

test('Column initialization', () => {
    const length = 100;
    const col = new Column({ type: TypeId.INT32, data: new Int32Buffer(length) });

    expect(col.type.id).toBe(TypeId.INT32);
    expect(col.length).toBe(length);
    expect(col.nullCount).toBe(0);
    expect(col.hasNulls).toBe(false);
    expect(col.nullable).toBe(false);
});

test('Column initialization with null_mask', () => {
    const length = 100;
    const col = new Column({
        type: TypeId.BOOL8,
        data: new Uint8Buffer(length),
        nullMask: new Uint8Buffer(64),
    });

    expect(col.type.id).toBe(TypeId.BOOL8);
    expect(col.length).toBe(length);
    expect(col.nullCount).toBe(100);
    expect(col.hasNulls).toBe(true);
    expect(col.nullable).toBe(true);
});

test('Column null_mask, null_count', () => {
    const length = 32;
    const col = new Column({
        type: TypeId.FLOAT32,
        data: new Float32Buffer(length),
        nullMask: new Uint8Buffer([254, 255, 255, 255])
    });

    expect(col.type.id).toBe(TypeId.FLOAT32);
    expect(col.length).toBe(length);
    expect(col.nullCount).toBe(1);
    expect(col.hasNulls).toBe(true);
    expect(col.nullable).toBe(true);
});

test('test child(child_index), num_children', () => {
    const utf8Col = new Column({ type: TypeId.UINT8, data: new Uint8Buffer(Buffer.from("hello")) });
    const offsetsCol = new Column({ type: TypeId.INT32, data: new Int32Buffer([0, utf8Col.length]) });
    const stringsCol = new Column({
        type: TypeId.STRING,
        length: 1,
        nullMask: new Uint8Buffer([255]),
        children: [offsetsCol, utf8Col],
    });

    expect(stringsCol.type.id).toBe(TypeId.STRING);
    expect(stringsCol.numChildren).toBe(2);
    expect(stringsCol[0]).toBe("hello");
    expect(stringsCol.getChild(0).length).toBe(offsetsCol.length);
    expect(stringsCol.getChild(0).type.id).toBe(offsetsCol.type.id);
    expect(stringsCol.getChild(1).length).toBe(utf8Col.length);
    expect(stringsCol.getChild(1).type.id).toBe(utf8Col.type.id);
});

// test('test Column(column) constructor', () => {
//     const buffer_size = 100;
//     const db = new DeviceBuffer(buffer_size,10);

//     const null_mask = new DeviceBuffer(buffer_size,10);
//     const col = new Column(types.FLOAT32, 10, db, null_mask, 1);
//     const col1 = new Column(col);

//     expect(col1.type.id).toBe(TypeId.FLOAT32);
//     expect(col1.length).toBe(buffer_size);
//     expect(col1.nullCount).toBe(1);
//     expect(col1.hasNulls).toBe(true);
//     expect(col1.nullable).toBe(true);
// });


test('Column argsort', () => {
    const arr = new Column({ type: TypeId.INT32, data: new Int32Buffer([1, 3, 5, 4, 2, 0]) });
    const result = arr.argsort();

    expect(result.type.id).toBe(TypeId.INT32);
    expect(result.length).toBe(arr.length);
    expect(result.nullCount).toBe(0);
    expect(result.hasNulls).toBe(false);
    expect(result.nullable).toBe(false);

    // TODO (bev) need a better way to do this. E.g. perhaps a to_array to copy
    // to host, then implement a helper similar to assert_eq on the python side?
    // https://github.com/rapidsai/cudf/blob/branch-0.18/python/cudf/cudf/tests/utils.py#L69
    const expected = [5, 0, 4, 1, 3, 2];
    for (const ind in expected) {
        expect(result[ind]).toBe(expected[ind])
    }
});