// Copyright (c) 2026, NVIDIA CORPORATION.
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

import {
  Int32Buffer,
  setDefaultAllocator,
} from '@rapidsai/cuda';
import {Int32, Series} from '@rapidsai/cudf';
import {DeviceBuffer} from '@rapidsai/rmm';

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength));

test('Series.getValue', () => {
  const col = Series.new({type: new Int32, data: new Int32Buffer([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])});
  for (let i = 0; i < 10; i++) { expect(col.getValue(i)).toEqual(i); }
});

test('Series.setValue', () => {
  const col = Series.new({type: new Int32, data: new Int32Buffer([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])});
  col.setValue(2, 999);
  col.setValue(4, 999);
  col.setValue(5, 999);
  col.setValue(8, 999);
  expect([...col]).toEqual([0, 1, 999, 3, 999, 999, 6, 7, 999, 9]);
});

test('Series.setValues (series)', () => {
  const col     = Series.new({type: new Int32, data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]});
  const values  = Series.new({type: new Int32, data: [200, 400, 500, 800]});
  const indices = Series.new({type: new Int32, data: [2, 4, 5, 8]});

  col.setValues(indices, values);

  expect([...col]).toEqual([0, 1, 200, 3, 400, 500, 6, 7, 800, 9]);
});

test('Series.setValues (scalar)', () => {
  const col     = Series.new({type: new Int32, data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]});
  const indices = Series.new({type: new Int32, data: [2, 4, 5, 8]});

  col.setValues(indices, 999);

  expect([...col]).toEqual([0, 1, 999, 3, 999, 999, 6, 7, 999, 9]);
});
