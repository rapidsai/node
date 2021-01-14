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

setDefaultAllocator((byteLength) => new DeviceBuffer(byteLength, mr));

test('Table initialization', () => {
  const length = 100;
  const col_0  = new Column({type: TypeId.INT32, data: new Int32Buffer(length)});

  const col_1   = new Column({
    type: TypeId.BOOL8,
    data: new Uint8Buffer(length),
    nullMask: new Uint8Buffer(64),
  });
  const table_0 = new Table({columns: [col_0, col_1]});
  expect(table_0.numColumns).toBe(2)
});

test('Table getColumnByIndex', () => {
  const length = 100;
  const col_0  = new Column({type: TypeId.INT32, data: new Int32Buffer(length)});

  const col_1   = new Column({
    type: TypeId.BOOL8,
    data: new Uint8Buffer(length),
    nullMask: new Uint8Buffer(64),
  });
  const table_0 = new Table({columns: [col_0, col_1]})

  expect(table_0.getColumnByIndex(0).type.id).toBe(TypeId.INT32);
  expect(() => { table_0.getColumnByIndex(4); }).toThrow();
});
