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

function testNumberAll<T extends Int8|Int16|Int32|Uint8|Uint16|Uint32|Float32|Float64>(
  type: T, data: (T['scalarType']|null)[], skipNulls = true) {
  const expected = skipNulls ? data.every((x) => x === null || x !== 0)  //
                             : data.every((x) => x !== null);
  expect(Series.new({type, data}).all(skipNulls)).toEqual(expected);
}

function testBigIntAll<T extends Int64|Uint64>(
  type: T, data: (T['scalarType']|null)[], skipNulls = true) {
  const expected = skipNulls ? data.every((x) => x === null || x !== 0n)  //
                             : data.every((x) => x !== null);
  expect(Boolean(Series.new({type, data}).all(skipNulls))).toEqual(expected);
}

function testBooleanAll<T extends Bool8>(
  type: T, data: (T['scalarType']|null)[], skipNulls = true) {
  const expected = skipNulls ? data.every((x) => x === null || x !== false)  //
                             : data.every((x) => x !== null);
  expect(Series.new({type, data}).all(skipNulls)).toEqual(expected);
}

describe.each([[true], [false]])('Series.all(skipNulls=%p)', (skipNulls) => {
  test('Int8', () => { testNumberAll(new Int8, numbers, skipNulls); });
  test('Int16', () => { testNumberAll(new Int16, numbers, skipNulls); });
  test('Int32', () => { testNumberAll(new Int32, numbers, skipNulls); });
  test('Int64', () => { testBigIntAll(new Int64, bigints, skipNulls); });
  test('Uint8', () => { testNumberAll(new Uint8, numbers, skipNulls); });
  test('Uint16', () => { testNumberAll(new Uint16, numbers, skipNulls); });
  test('Uint32', () => { testNumberAll(new Uint32, numbers, skipNulls); });
  test('Uint64', () => { testBigIntAll(new Uint64, bigints, skipNulls); });
  test('Float32', () => { testNumberAll(new Float32, numbers, skipNulls); });
  test('Float64', () => { testNumberAll(new Float64, numbers, skipNulls); });
  test('Bool8', () => { testBooleanAll(new Bool8, bools, skipNulls); });
  test('Float32', () => { testNumberAll(new Float32, float_with_NaN, skipNulls); });
  test('Float64', () => { testNumberAll(new Float64, float_with_NaN, skipNulls); });
});
