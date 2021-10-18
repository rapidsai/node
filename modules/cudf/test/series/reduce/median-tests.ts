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

function jsMedian(values: any) {
  if (values.length === 0) return NaN;
  values.sort(function(a: any, b: any) { return Number(a) - Number(b); });
  const half = Math.floor(values.length / 2);
  if (values.length % 2) return Number(values[half]);
  return (Number(values[half - 1]) + Number(values[half])) / 2.0;
}

function testNumberMedian<T extends Int8|Int16|Int32|Uint8|Uint16|Uint32|Float32|Float64>(
  type: T, data: (T['scalarType']|null)[], skipNulls = true) {
  if (skipNulls) {
    const expected = jsMedian(data.filter((x) => x !== null && !isNaN(x)));
    expect(Series.new({type, data}).median(skipNulls)).toEqual(expected);
  } else {
    const expected = data.some((x) => x === null || isNaN(x)) ? NaN : jsMedian(data);
    expect(Series.new({type, data}).median(skipNulls)).toEqual(expected);
  }
}

function testBigIntMedian<T extends Int64|Uint64>(
  type: T, data: (T['scalarType']|null)[], skipNulls = true) {
  if (skipNulls) {
    const expected = jsMedian(data.filter((x) => x !== null));
    expect(Series.new({type, data}).median(skipNulls)).toEqual(expected);
  } else {
    const expected = data.some((x) => x === null) ? NaN : jsMedian(data);
    expect(Series.new({type, data}).median(skipNulls)).toEqual(expected);
  }
}

function testBooleanMedian<T extends Bool8>(
  type: T, data: (T['scalarType']|null)[], skipNulls = true) {
  if (skipNulls) {
    const expected = jsMedian(data.filter((x) => x !== null));
    expect(Series.new({type, data}).median(skipNulls)).toEqual(expected);
  } else {
    const expected = data.some((x) => x === null) ? NaN : jsMedian(data);
    expect(Series.new({type, data}).median(skipNulls)).toEqual(expected);
  }
}

describe.each([[true], [false]])('Series.any(skipNulls=%p)', (skipNulls) => {
  test('Int8', () => { testNumberMedian(new Int8, numbers, skipNulls); });
  test('Int16', () => { testNumberMedian(new Int16, numbers, skipNulls); });
  test('Int32', () => { testNumberMedian(new Int32, numbers, skipNulls); });
  test('Int64', () => { testBigIntMedian(new Int64, bigints, skipNulls); });
  test('Uint8', () => { testNumberMedian(new Uint8, numbers, skipNulls); });
  test('Uint16', () => { testNumberMedian(new Uint16, numbers, skipNulls); });
  test('Uint32', () => { testNumberMedian(new Uint32, numbers, skipNulls); });
  test('Uint64', () => { testBigIntMedian(new Uint64, bigints, skipNulls); });
  test('Float32', () => { testNumberMedian(new Float32, numbers, skipNulls); });
  test('Float64', () => { testNumberMedian(new Float64, numbers, skipNulls); });
  test('Bool8', () => { testBooleanMedian(new Bool8, bools, skipNulls); });
  test('Float32', () => { testNumberMedian(new Float32, float_with_NaN, skipNulls); });
  test('Float64', () => { testNumberMedian(new Float64, float_with_NaN, skipNulls); });
});
