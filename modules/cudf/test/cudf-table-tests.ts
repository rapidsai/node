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

import {Int32Buffer, setDefaultAllocator, Uint8Buffer} from '@nvidia/cuda';
import {Column, Table, TypeId} from '@nvidia/cudf';
import {CudaMemoryResource, DeviceBuffer} from '@nvidia/rmm';

const mr = new CudaMemoryResource();

setDefaultAllocator((byteLength) => new DeviceBuffer(byteLength, 0, mr));

test('Table initialization', () => {
  const length = 100;
  const col_0  = new Column({type: TypeId.INT32, data: new Int32Buffer(length)});

  const col_1   = new Column({
    type: TypeId.BOOL8,
    data: new Uint8Buffer(length),
    nullMask: new Uint8Buffer(64),
  });
  const table_0 = new Table({data: {"col_0": col_0, "col_1": col_1}});
  expect(table_0.numColumns).toBe(2);
  expect(table_0.numRows).toBe(length);
  expect(table_0.columns).toStrictEqual(['col_0', 'col_1']);
  expect(table_0 ["col_0"].type.id).toBe(col_0.type.id);
  expect(table_0 ["col_1"].type.id).toBe(col_1.type.id);
});

test('Table getColumn', () => {
  const length = 100;
  const col_0  = new Column({type: TypeId.INT32, data: new Int32Buffer(length)});

  const col_1   = new Column({
    type: TypeId.BOOL8,
    data: new Uint8Buffer(length),
    nullMask: new Uint8Buffer(64),
  });
  const table_0 = new Table({data: {"col_0": col_0, "col_1": col_1}});
  expect(table_0.getColumnByName("col_0").type.id).toBe(col_0.type.id);
  expect(table_0.getColumnByIndex(1).type.id).toBe(col_1.type.id);
  expect(() => { table_0.getColumnByIndex(2); }).toThrow();
  expect(() => { table_0.getColumnByName(2); }).toThrow();

  expect(() => { table_0.getColumnByName("junk"); }).toThrow();
});

test('Table.select', () => {
  const length = 100;
  const col_0  = new Column({type: TypeId.INT32, data: new Int32Buffer(length)});

  const col_1 = new Column({
    type: TypeId.BOOL8,
    data: new Uint8Buffer(length),
    nullMask: new Uint8Buffer(64),
  });

  const col_2 = new Column({type: TypeId.INT32, data: new Int32Buffer(length)});
  const col_3 = new Column({type: TypeId.INT32, data: new Int32Buffer(length)});

  const table_0 =
    new Table({data: {"col_0": col_0, "col_1": col_1, "col_2": col_2, "col_3": col_3}});

  expect(table_0.numColumns).toBe(4);
  expect(table_0.numRows).toBe(length);
  expect(table_0.columns).toStrictEqual(["col_0", "col_1", "col_2", "col_3"]);

  expect(table_0.select(["col_0"])).toStrictEqual(new Table({data: {"col_0": col_0}}));
  expect(table_0.select(["col_0", "col_3"])).toStrictEqual(new Table({
    data: {"col_0": col_0, "col_3": col_3}
  }));
});

test('Table.slice', () => {
  const length = 100;
  const col_0  = new Column({type: TypeId.INT32, data: new Int32Buffer(length)});

  const col_1 = new Column({
    type: TypeId.BOOL8,
    data: new Uint8Buffer(length),
    nullMask: new Uint8Buffer(64),
  });

  const col_2 = new Column({type: TypeId.INT32, data: new Int32Buffer(length)});
  const col_3 = new Column({type: TypeId.INT32, data: new Int32Buffer(length)});

  const table_0 =
    new Table({data: {"col_0": col_0, "col_1": col_1, "col_2": col_2, "col_3": col_3}});

  expect(table_0.numColumns).toBe(4);
  expect(table_0.numRows).toBe(length);
  expect(table_0.columns).toStrictEqual(["col_0", "col_1", "col_2", "col_3"]);

  expect(table_0.slice(2, 3)).toStrictEqual(new Table({data: {"col_2": col_0, "col_3": col_3}}));
  expect(table_0.slice("col_1", "col_3")).toStrictEqual(new Table({
    data: {"col_1": col_1, "col_2": col_2, "col_3": col_3}
  }));
});

test('Table addColumn and drop', () => {
  const length = 100;
  const col_0  = new Column({type: TypeId.INT32, data: new Int32Buffer(length)});

  const col_1 = new Column({
    type: TypeId.BOOL8,
    data: new Uint8Buffer(length),
    nullMask: new Uint8Buffer(64),
  });

  const col_2 = new Column({type: TypeId.INT32, data: new Int32Buffer(length)});
  const col_3 = new Column({type: TypeId.INT32, data: new Int32Buffer(length)});

  const table_0 = new Table({data: {"col_0": col_0, "col_1": col_1, "col_2": col_2}});

  table_0.addColumn("col_3", col_3);
  expect(table_0.numColumns).toBe(4);
  expect(table_0.numRows).toBe(length);
  expect(table_0.columns).toStrictEqual(["col_0", "col_1", "col_2", "col_3"]);

  table_0.drop({columns: ["col_1"]});
  expect(table_0.numColumns).toBe(3);
  expect(table_0.numRows).toBe(length);
  expect(table_0.columns).toStrictEqual(["col_0", "col_2", "col_3"]);
});
