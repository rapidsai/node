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
} from '@nvidia/cudf';
import {DeviceBuffer} from '@nvidia/rmm';

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength));

const numbers        = [null, 0, 1, 1, null, 2, 3, 3, 4, 4];
const bigints        = [null, 0n, 1n, 1n, null, 2n, 3n, 3n, 4n, 4n];
const bools          = [null, false, true, true, null, true, false, true, false, true];
const float_with_NaN = [NaN, 1, 2, 3, 4, 3, 7, 7, 2, NaN];

function testNumberAny<T extends Int8|Int16|Int32|Uint8|Uint16|Uint32|Float32|Float64>(
  type: T, data: (T['scalarType']|null)[], skipna = true) {
  const expected = skipna ? data.some((x) => x === null || x !== 0)  //
                          : data.some((x) => x !== null);
  expect(Series.new({type, data}).any(skipna)).toEqual(expected);
}

function testBigIntAny<T extends Int64|Uint64>(
  type: T, data: (T['scalarType']|null)[], skipna = true) {
  const expected = skipna ? data.some((x) => x === null || x !== 0n)  //
                          : data.some((x) => x !== null);
  expect(Boolean(Series.new({type, data}).any(skipna))).toEqual(expected);
}

function testBooleanAny<T extends Bool8>(type: T, data: (T['scalarType']|null)[], skipna = true) {
  const expected = skipna ? data.some((x) => x === null || x !== false)  //
                          : data.some((x) => x !== null);
  expect(Series.new({type, data}).any(skipna)).toEqual(expected);
}

describe.each([[true], [false]])('Series.any(skipna=%p)', (skipna) => {
  test('Int8', () => { testNumberAny(new Int8, numbers, skipna); });
  test('Int16', () => { testNumberAny(new Int16, numbers, skipna); });
  test('Int32', () => { testNumberAny(new Int32, numbers, skipna); });
  test('Int64', () => { testBigIntAny(new Int64, bigints, skipna); });
  test('Uint8', () => { testNumberAny(new Uint8, numbers, skipna); });
  test('Uint16', () => { testNumberAny(new Uint16, numbers, skipna); });
  test('Uint32', () => { testNumberAny(new Uint32, numbers, skipna); });
  test('Uint64', () => { testBigIntAny(new Uint64, bigints, skipna); });
  test('Float32', () => { testNumberAny(new Float32, numbers, skipna); });
  test('Float64', () => { testNumberAny(new Float64, numbers, skipna); });
  test('Bool8', () => { testBooleanAny(new Bool8, bools, skipna); });
  test('Float32', () => { testNumberAny(new Float32, float_with_NaN, skipna); });
  test('Float64', () => { testNumberAny(new Float64, float_with_NaN, skipna); });
});
