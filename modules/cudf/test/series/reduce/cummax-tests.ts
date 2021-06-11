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

const numbers        = [4, 2, null, 5, 1, 1];
const float_with_NaN = [4, 2, NaN, 5, 1, 1];
const bools          = [true, false, null, false, true];
const bigints        = [4n, 2n, null, 5n, 1n, 1n];

function testNumberAny<T extends Int8|Int16|Int32|Uint8|Uint16|Uint32|Float32|Float64>(
  type: T, data: (T['scalarType']|null)[], skipna = true) {
  const expected = skipna ? [4, 4, null, 5, 5, 5]  //
                          : [4, 4, null, null, null, null];
  expect([...Series.new({type, data}).cummax(skipna)]).toEqual(expected);
}

function testBigIntAny<T extends Int64|Uint64>(
  type: T, data: (T['scalarType']|null)[], skipna = true) {
  const expected = skipna ? [4n, 4n, null, 5n, 5n, 5n]  //
                          : [4n, 4n, null, null, null, null];
  expect([...Series.new({type, data}).cummax(skipna)]).toEqual(expected);
}

function testBooleanAny<T extends Bool8>(type: T, data: (T['scalarType']|null)[], skipna = true) {
  const expected = skipna ? [true, true, null, true, true]  //
                          : [true, true, null, null, null];

  expect([...Series.new({type, data}).cummax(skipna)]).toEqual(expected);
}

describe.each([[true], [false]])('Series.cummax(skipna=%p)', (skipna) => {
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
  test('Float32-nan', () => { testNumberAny(new Float32, float_with_NaN, skipna); });
  test('Float64-nan', () => { testNumberAny(new Float64, float_with_NaN, skipna); });
});
