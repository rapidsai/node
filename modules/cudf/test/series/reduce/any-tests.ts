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

import {BigIntArray, setDefaultAllocator, TypedArray, Uint8Buffer} from '@nvidia/cuda';
import {
  Bool8,
  Float32,
  Float64,
  Int16,
  Int32,
  Int64,
  Int8,
  Numeric,
  Series,
  Uint16,
  Uint32,
  Uint64,
  Uint8
} from '@nvidia/cudf';
import {DeviceBuffer} from '@nvidia/rmm';
import {BoolVector} from "apache-arrow";

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength));

const numbers = Array.from([null, 0, 1, 1, null, 2, 3, 3, 4, 4]);

const makeBigInts = (length = 10) => Array.from({length}, (_, i) => BigInt(parseInt(i / 3)));

const makeBooleans = (length = 10) => Array.from({length}, (_, i) => Number(i % 2 == 0));

const float_with_NaN = Array.from([NaN, 1, 2, 3, 4, 3, 7, 7, 2, NaN]);

function js_any(values: any) { return values.some(x => x != false); }

function testNumberany<T extends Numeric, R extends TypedArray>(type: T, data: R) {
  expect(Series.new({type, data}).any()).toEqual(js_any(data));
}

function testNumberanySkipNA<T extends Numeric, R extends TypedArray>(
  type: T, data: R, mask_array: Array<number>) {
  const mask = new Uint8Buffer(BoolVector.from(mask_array).values);
  // skipna=false
  expect(Series.new({type, data, nullMask: mask}).any(false))
    .toEqual(js_any(data.filter((_, i) => mask_array[i] == 1)));
}

function testBigIntany<T extends Numeric, R extends BigIntArray>(type: T, data: R) {
  expect(Series.new({type, data}).any()).toEqual(js_any(data));
}

function testBigIntanySkipNA<T extends Numeric, R extends BigIntArray>(
  type: T, data: R, mask_array: Array<number>) {
  const mask = new Uint8Buffer(BoolVector.from(mask_array).values);
  // skipna=false
  expect(Series.new({type, data, nullMask: mask}).any(false))
    .toEqual(js_any(data.filter((_, i) => mask_array[i] == 1)));
}

describe('Series.any(skipna=true)', () => {
  test('Int8', () => { testNumberany(new Int8, new Int8Array(numbers)); });
  test('Int16', () => { testNumberany(new Int16, new Int16Array(numbers)); });
  test('Int32', () => { testNumberany(new Int32, new Int32Array(numbers)); });
  test('Int64', () => { testBigIntany(new Int64, new BigInt64Array(makeBigInts())); });
  test('Uint8', () => { testNumberany(new Uint8, new Uint8Array(numbers)); });
  test('Uint16', () => { testNumberany(new Uint16, new Uint16Array(numbers)); });
  test('Uint32', () => { testNumberany(new Uint32, new Uint32Array(numbers)); });
  test('Uint64', () => { testBigIntany(new Uint64, new BigUint64Array(makeBigInts())); });
  test('Float32', () => { testNumberany(new Float32, new Float32Array(numbers)); });
  test('Float64', () => { testNumberany(new Float64, new Float64Array(numbers)); });
  test('Bool8', () => { testNumberany(new Bool8, new Uint8ClampedArray(makeBooleans())); });
});

describe("Series.any(skipna=false)", () => {
  test('Int8', () => {testNumberanySkipNA(new Int8, new Int8Array(numbers), makeBooleans())});
  test('Int16', () => { testNumberanySkipNA(new Int16, new Int16Array(numbers), makeBooleans()); });
  test('Int32', () => { testNumberanySkipNA(new Int32, new Int32Array(numbers), makeBooleans()); });
  test('Int64',
       () => { testBigIntanySkipNA(new Int64, new BigInt64Array(makeBigInts()), makeBooleans()); });
  test('Uint8', () => { testNumberanySkipNA(new Uint8, new Uint8Array(numbers), makeBooleans()); });
  test('Uint16',
       () => { testNumberanySkipNA(new Uint16, new Uint16Array(numbers), makeBooleans()); });
  test('Uint32',
       () => { testNumberanySkipNA(new Uint32, new Uint32Array(numbers), makeBooleans()); });
  test(
    'Uint64',
    () => { testBigIntanySkipNA(new Uint64, new BigUint64Array(makeBigInts()), makeBooleans()); });
  test('Float32',
       () => { testNumberanySkipNA(new Float32, new Float32Array(numbers), makeBooleans()); });
  test('Float64',
       () => { testNumberanySkipNA(new Float64, new Float64Array(numbers), makeBooleans()); });
  test('Bool8', () => {
    testNumberanySkipNA(new Bool8, new Uint8ClampedArray(makeBooleans()), makeBooleans());
  });
});

describe("Float type Series with NaN => Series.any(skipna=true)", () => {
  test('Float32', () => { testNumberany(new Float32, new Float32Array(float_with_NaN)); });
  test('Float64', () => { testNumberany(new Float64, new Float64Array(float_with_NaN)); });
});

describe("Float type Series with NaN => Series.any(skipna=false)", () => {
  test(
    'Float32',
    () => { testNumberanySkipNA(new Float32, new Float32Array(float_with_NaN), makeBooleans()); });
  test(
    'Float64',
    () => { testNumberanySkipNA(new Float64, new Float64Array(float_with_NaN), makeBooleans()); });
});
