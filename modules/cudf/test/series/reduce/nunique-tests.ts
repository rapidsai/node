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
  Bool8,
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

const numbers        = [null, 0, 1, 1, null, 2, 3, 3, 4, 4];
const bigints        = [null, 0n, 1n, 1n, null, 2n, 3n, 3n, 4n, 4n];
const bools          = [null, false, true, true, null, true, false, true, false, true];
const float_with_NaN = [NaN, 1, 2, 3, 4, 3, 7, 7, 2, NaN];

function testNumberNunique<T extends Int8|Int16|Int32|Uint8|Uint16|Uint32|Float32|Float64>(
  type: T, data: (T['scalarType']|null)[], expected: number, dropna = true) {
  expect(Series.new({type, data}).nunique(dropna)).toEqual(expected + (dropna ? 0 : 1));
}

function testBigIntNunique<T extends Int64|Uint64>(
  type: T, data: (T['scalarType']|null)[], expected: number, dropna = true) {
  expect(Series.new({type, data}).nunique(dropna)).toEqual(expected + (dropna ? 0 : 1));
}

function testBooleanNunique<T extends Bool8>(
  type: T, data: (T['scalarType']|null)[], expected: number, dropna = true) {
  expect(Series.new({type, data}).nunique(dropna)).toEqual(expected + (dropna ? 0 : 1));
}

describe.each([[true], [false]])('Series.any(dropna=%p)', (dropna) => {
  test('Int8', () => { testNumberNunique(new Int8, numbers, 5, dropna); });
  test('Int16', () => { testNumberNunique(new Int16, numbers, 5, dropna); });
  test('Int32', () => { testNumberNunique(new Int32, numbers, 5, dropna); });
  test('Int64', () => { testBigIntNunique(new Int64, bigints, 5, dropna); });
  test('Uint8', () => { testNumberNunique(new Uint8, numbers, 5, dropna); });
  test('Uint16', () => { testNumberNunique(new Uint16, numbers, 5, dropna); });
  test('Uint32', () => { testNumberNunique(new Uint32, numbers, 5, dropna); });
  test('Uint64', () => { testBigIntNunique(new Uint64, bigints, 5, dropna); });
  test('Float32', () => { testNumberNunique(new Float32, numbers, 5, dropna); });
  test('Float64', () => { testNumberNunique(new Float64, numbers, 5, dropna); });
  test('Bool8', () => { testBooleanNunique(new Bool8, bools, 2, dropna); });
  test('Float32', () => { testNumberNunique(new Float32, float_with_NaN, 5, dropna); });
  test('Float64', () => { testNumberNunique(new Float64, float_with_NaN, 5, dropna); });
});
