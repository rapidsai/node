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

import '../../jest-extensions';

import {setDefaultAllocator} from '@rapidsai/cuda';
import {
  Float32,
  Float64,
  Int16,
  Int32,
  Int64,
  Int8,
  Series,
  Uint16,
  Uint32,
  Uint64,
  Uint8
} from '@rapidsai/cudf';
import {DeviceBuffer} from '@rapidsai/rmm';

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength));

const numbers        = [1, null, 2, 3, 4, 5, null, 10];
const bigints        = [1n, null, 2n, 3n, 4n, 5n, null, 10n];
const float_with_NaN = [1.0, NaN, 2.0, 3.0, 4.0, 5.0, NaN, 10.0];

const result = 2.4377317925288935;

function testNumberKurtosis<T extends Int8|Int16|Int32|Int64|Uint8|Uint16|Uint32|Uint64|Float32|
                            Float64>(type: T, data: (T['scalarType']|null)[], skipNulls = true) {
  const expected = skipNulls ? result : NaN;
  expect(Series.new({type, data}).kurtosis(skipNulls)).toEqual(expected);
}

describe.each([[true], [false]])('Series.kurtosis(skipNulls=%p)', (skipNulls) => {
  test('Int8', () => { testNumberKurtosis(new Int8, numbers, skipNulls); });
  test('Int16', () => { testNumberKurtosis(new Int16, numbers, skipNulls); });
  test('Int32', () => { testNumberKurtosis(new Int32, numbers, skipNulls); });
  test('Int64', () => { testNumberKurtosis(new Int64, bigints, skipNulls); });
  test('Uint8', () => { testNumberKurtosis(new Uint8, numbers, skipNulls); });
  test('Uint16', () => { testNumberKurtosis(new Uint16, numbers, skipNulls); });
  test('Uint32', () => { testNumberKurtosis(new Uint32, numbers, skipNulls); });
  test('Uint64', () => { testNumberKurtosis(new Uint64, bigints, skipNulls); });
  test('Float32', () => { testNumberKurtosis(new Float32, numbers, skipNulls); });
  test('Float64', () => { testNumberKurtosis(new Float64, numbers, skipNulls); });
  test('Float32-nan', () => { testNumberKurtosis(new Float32, float_with_NaN, skipNulls); });
  test('Float64-nan', () => { testNumberKurtosis(new Float64, float_with_NaN, skipNulls); });
});

describe.each([[[]], [[2]], [[2, 3]], [[2, 3, -1]]])('Too short (data=%p)', (data) => {
  test('returns NaN',
       () => { expect(Series.new({type: new Float32, data: data}).kurtosis()).toBe(NaN); });
});

describe('Zero variance', () => {
  test('returns 0', () => {
    expect(Series.new({type: new Float32, data: [1, 1, 1, 1, 1, 1]}).kurtosis()).toBe(0);
  });
});
