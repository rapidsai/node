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

import {Float32Buffer, Int32Buffer, setDefaultAllocator, Uint8Buffer} from '@nvidia/cuda';
import {
  Bool8,
  Column,
  Float32,
  Int32,
  Table,
} from '@rapidsai/cudf';
import {CudaMemoryResource, DeviceBuffer} from '@rapidsai/rmm';
import * as arrow from 'apache-arrow';

const mr = new CudaMemoryResource();

setDefaultAllocator((byteLength) => new DeviceBuffer(byteLength, mr));

test('Table initialization', () => {
  const length = 100;
  const col_0  = new Column({type: new Int32, data: new Int32Buffer(length)});

  const col_1   = new Column({
    type: new Bool8,
    data: new Uint8Buffer(length),
    nullMask: new Uint8Buffer(64),
  });
  const table_0 = new Table({columns: [col_0, col_1]});
  expect(table_0.numColumns).toBe(2)
});

test('Table.getColumnByIndex', () => {
  const length = 100;
  const col_0  = new Column({type: new Int32, data: new Int32Buffer(length)});

  const col_1   = new Column({
    type: new Bool8,
    data: new Uint8Buffer(length),
    nullMask: new Uint8Buffer(64),
  });
  const table_0 = new Table({columns: [col_0, col_1]})

  expect(table_0.getColumnByIndex(0).type).toBeInstanceOf(Int32);
  expect(() => { table_0.getColumnByIndex(4); }).toThrow();
});

test('Table.gather (bad argument)', () => {
  const col_0   = new Column({type: new Int32, data: new Int32Buffer([0, 1, 2, 3, 4, 5])});
  const table_0 = new Table({columns: [col_0]});

  const selection = [2, 4, 5];

  expect(() => table_0.gather(<any>selection)).toThrow();
})

test('Table.gather (indices)', () => {
  const col_0 = new Column({type: new Int32, data: new Int32Buffer([0, 1, 2, 3, 4, 5])});
  const col_1 =
    new Column({type: new Float32, data: new Float32Buffer([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])});
  const table_0 = new Table({columns: [col_0, col_1]})

  const selection = new Column({type: new Int32, data: new Int32Buffer([2, 4, 5])});

  const result = table_0.gather(selection);
  expect(result.numRows).toBe(3);

  const r0 = result.getColumnByIndex(0);
  const r1 = result.getColumnByIndex(1);

  expect(r0.type.typeId).toBe(arrow.Type.Int);
  expect(r0.type.bitWidth).toBe(32);
  expect(r0.getValue(0)).toBe(2);
  expect(r0.getValue(1)).toBe(4);
  expect(r0.getValue(2)).toBe(5);

  expect(r1.type.typeId).toBe(arrow.Type.Float);
  expect(r1.type.precision).toBe(arrow.Precision.SINGLE);
  expect(r1.getValue(0)).toBe(2.0);
  expect(r1.getValue(1)).toBe(4.0);
  expect(r1.getValue(2)).toBe(5.0);
});

test('Table.gather (bitmask)', () => {
  const col_0 = new Column({type: new Int32, data: new Int32Buffer([0, 1, 2, 3, 4, 5])});
  const col_1 =
    new Column({type: new Float32, data: new Float32Buffer([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])});
  const table_0 = new Table({columns: [col_0, col_1]})

  const selection = new Column({
    length: 6,
    type: new Bool8,
    data: new Uint8Buffer([0, 0, 1, 0, 1, 1]),
  });

  const result = table_0.gather(selection);
  expect(result.numRows).toBe(3);

  const r0 = result.getColumnByIndex(0);
  const r1 = result.getColumnByIndex(1);

  expect(r0.type.typeId).toBe(arrow.Type.Int);
  expect(r0.type.bitWidth).toBe(32);
  expect(r0.getValue(0)).toBe(2);
  expect(r0.getValue(1)).toBe(4);
  expect(r0.getValue(2)).toBe(5);

  expect(r1.type.typeId).toBe(arrow.Type.Float);
  expect(r1.type.precision).toBe(arrow.Precision.SINGLE);
  expect(r1.getValue(0)).toBe(2.0);
  expect(r1.getValue(1)).toBe(4.0);
  expect(r1.getValue(2)).toBe(5.0);
});
