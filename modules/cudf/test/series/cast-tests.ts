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

import {Float32Buffer, Int32Buffer, setDefaultAllocator} from '@rapidsai/cuda';
import {Float32, Float64, Int32, Int64, Series} from '@rapidsai/cudf';
import {DeviceBuffer} from '@rapidsai/rmm';

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength));

test('NumericSeries.cast', () => {
  const col = Series.new({type: new Int32, data: new Int32Buffer([0, 1, 2, 3, 4])});

  expect(col.cast(new Int64).type).toBeInstanceOf(Int64);
  expect(col.cast(new Float32).type).toBeInstanceOf(Float32);
  expect(col.cast(new Float64).type).toBeInstanceOf(Float64);

  const floatCol = Series.new({type: new Float32, data: new Float32Buffer([1.5, 2.8, 3.1, 4.2])});
  const result   = floatCol.cast(new Int32);

  expect([...result]).toEqual([1, 2, 3, 4]);
});
