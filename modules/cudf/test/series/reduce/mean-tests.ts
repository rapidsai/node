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

function jsMean(values: number[]) {
  if (values.length === 0) return NaN;
  return values.reduce((x: number, y: number) => x + y) / values.length;
}

function jsMeanBigInt(values: bigint[]) {
  if (values.length === 0) return NaN;
  return Number(values.reduce((x: bigint, y: bigint) => x + y)) / values.length;
}

function jsMeanBoolean(values: boolean[]) {
  if (values.length === 0) return NaN;
  let sum = 0;
  values.forEach((x) => sum += (x ? 1 : 0));
  return sum / values.length;
}

function testNumberMean<T extends Int8|Int16|Int32|Uint8|Uint16|Uint32|Float32|Float64>(
  type: T, data: (T['scalarType']|null)[], skipna = true) {
  if (skipna) {
    const expected = jsMean(data.filter((x) => x !== null && !isNaN(x)) as number[]);
    expect(Series.new({type, data}).mean(skipna)).toEqual(expected);
  } else {
    const expected = data.some((x) => x === null || isNaN(x)) ? NaN : jsMean(data as number[]);
    expect(Series.new({type, data}).mean(skipna)).toEqual(expected);
  }
}

function testBigIntMean<T extends Int64|Uint64>(
  type: T, data: (T['scalarType']|null)[], skipna = true) {
  if (skipna) {
    const expected = jsMeanBigInt(data.filter((x) => x !== null) as bigint[]);
    expect(Series.new({type, data}).mean(skipna)).toEqual(expected);
  } else {
    const expected = data.some((x) => x === null) ? NaN : jsMeanBigInt(data as bigint[]);
    expect(Series.new({type, data}).mean(skipna)).toEqual(expected);
  }
}

function testBooleanMean<T extends Bool8>(type: T, data: (T['scalarType']|null)[], skipna = true) {
  if (skipna) {
    const expected = jsMeanBoolean(data.filter((x) => x !== null) as boolean[]);
    expect(Series.new({type, data}).mean(skipna)).toEqual(expected);
  } else {
    const expected = data.some((x) => x === null) ? NaN : jsMeanBoolean(data as boolean[]);
    expect(Series.new({type, data}).mean(skipna)).toEqual(expected);
  }
}

describe.each([[true], [false]])('Series.mean(skipna=%p)', (skipna) => {
  test('Int8', () => { testNumberMean(new Int8, numbers, skipna); });
  test('Int16', () => { testNumberMean(new Int16, numbers, skipna); });
  test('Int32', () => { testNumberMean(new Int32, numbers, skipna); });
  test('Int64', () => { testBigIntMean(new Int64, bigints, skipna); });
  test('Uint8', () => { testNumberMean(new Uint8, numbers, skipna); });
  test('Uint16', () => { testNumberMean(new Uint16, numbers, skipna); });
  test('Uint32', () => { testNumberMean(new Uint32, numbers, skipna); });
  test('Uint64', () => { testBigIntMean(new Uint64, bigints, skipna); });
  test('Float32', () => { testNumberMean(new Float32, numbers, skipna); });
  test('Float64', () => { testNumberMean(new Float64, numbers, skipna); });
  test('Bool8', () => { testBooleanMean(new Bool8, bools, skipna); });
  test('Float32', () => { testNumberMean(new Float32, float_with_NaN, skipna); });
  test('Float64', () => { testNumberMean(new Float64, float_with_NaN, skipna); });
});
