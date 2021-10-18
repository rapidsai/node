// Copyright (c) 2021, NVIDIA CORPORATION.
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

import {Float64Buffer, Int32Buffer, setDefaultAllocator} from '@rapidsai/cuda';
import {DataFrame, Float64, Int32, Series} from '@rapidsai/cudf';
import {DeviceBuffer} from '@rapidsai/rmm';

import {makeCSVString, toStringAsync} from './utils';

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength));

describe('DataFrame.writeCSV', () => {
  test('writes a CSV', async () => {
    const rows = [
      {a: 0, b: '1.0', c: '2'},
      {a: 1, b: '2.0', c: '3'},
      {a: 2, b: '3.0', c: '4'},
    ];
    const df = new DataFrame({
      a: Series.new({length: 3, type: new Int32, data: new Int32Buffer([0, 1, 2])}),
      b: Series.new({length: 3, type: new Float64, data: new Float64Buffer([1.0, 2.0, 3.0])}),
      c: Series.new(['2', '3', '4']),
    });
    expect((await toStringAsync(df.toCSV()))).toEqual(makeCSVString({rows}));
  });
});
