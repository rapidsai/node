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

import {Int32Buffer, setDefaultAllocator} from '@nvidia/cuda';
import {Column, Int32, Series, TypeId} from '@nvidia/cudf';
import {CudaMemoryResource, DeviceBuffer} from '@nvidia/rmm';

const mr = new CudaMemoryResource();

setDefaultAllocator((byteLength) => new DeviceBuffer(byteLength, 0, mr));

test('Series initialization with properties', () => {
  const length = 100;
  const s      = new Series({type: new Int32(), data: new Int32Buffer(length)});

  expect(s.type.id).toBe(TypeId.INT32);
  expect(s.length).toBe(length);
  expect(s.nullCount).toBe(0);
  expect(s.hasNulls).toBe(false);
  expect(s.nullable).toBe(false);
});

test('Series initialization with Column', () => {
  const length = 100;
  const col    = new Column({type: TypeId.INT32, data: new Int32Buffer(length)});
  const s      = new Series(col)

  expect(s.type.id).toBe(TypeId.INT32);
  expect(s.length).toBe(length);
  expect(s.nullCount).toBe(0);
  expect(s.hasNulls).toBe(false);
  expect(s.nullable).toBe(false);
});
