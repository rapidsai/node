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

import {setDefaultAllocator} from '@nvidia/cuda';
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

const result = 1.439590274527955;

function testNumberSkew<T extends Int8|Int16|Int32|Int64|Uint8|Uint16|Uint32|Uint64|Float32|
                        Float64>(type: T, data: (T['scalarType']|null)[], skipna = true) {
  const expected = skipna ? result : NaN;
  expect(Series.new({type, data}).skew(skipna)).toEqual(expected);
}

describe.each([[true], [false]])('Series.kurtosis(skipna=%p)', (skipna) => {
  test('Int8', () => { testNumberSkew(new Int8, numbers, skipna); });
  test('Int16', () => { testNumberSkew(new Int16, numbers, skipna); });
  test('Int32', () => { testNumberSkew(new Int32, numbers, skipna); });
  test('Int64', () => { testNumberSkew(new Int64, bigints, skipna); });
  test('Uint8', () => { testNumberSkew(new Uint8, numbers, skipna); });
  test('Uint16', () => { testNumberSkew(new Uint16, numbers, skipna); });
  test('Uint32', () => { testNumberSkew(new Uint32, numbers, skipna); });
  test('Uint64', () => { testNumberSkew(new Uint64, bigints, skipna); });
  test('Float32', () => { testNumberSkew(new Float32, numbers, skipna); });
  test('Float64', () => { testNumberSkew(new Float64, numbers, skipna); });
  test('Float32-nan', () => { testNumberSkew(new Float32, float_with_NaN, skipna); });
  test('Float64-nan', () => { testNumberSkew(new Float64, float_with_NaN, skipna); });
});

describe.each([[[]], [[2]], [[2, 3]]])('Too short (data=%p)', (data) => {
  test('returns NaN',
       () => { expect(Series.new({type: new Float32, data: data}).skew()).toBe(NaN); });
});

describe('Zero variance', () => {
  test('returns 0', () => {
    expect(Series.new({type: new Float32, data: [1, 1, 1, 1, 1, 1]}).skew()).toBe(0);
  });
});
