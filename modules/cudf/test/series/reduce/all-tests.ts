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

function js_all(values: any) { return values.every(x => x != false); }

function testNumberall<T extends Numeric, R extends TypedArray>(type: T, data: R) {
  expect(Series.new({type, data}).all()).toEqual(js_all(data));
}

function testNumberallSkipNA<T extends Numeric, R extends TypedArray>(
  type: T, data: R, mask_array: Array<number>) {
  const mask = new Uint8Buffer(BoolVector.from(mask_array).values);
  // skipna=false
  expect(Series.new({type, data, nullMask: mask}).all(false))
    .toEqual(js_all(data.filter((_, i) => mask_array[i] == 1)));
}

function testBigIntall<T extends Numeric, R extends BigIntArray>(type: T, data: R) {
  expect(Series.new({type, data}).all()).toEqual(js_all(data));
}

function testBigIntallSkipNA<T extends Numeric, R extends BigIntArray>(
  type: T, data: R, mask_array: Array<number>) {
  const mask = new Uint8Buffer(BoolVector.from(mask_array).values);
  // skipna=false
  expect(Series.new({type, data, nullMask: mask}).all(false))
    .toEqual(js_all(data.filter((_, i) => mask_array[i] == 1)));
}

describe('Series.all(skipna=true)', () => {
  test('Int8', () => { testNumberall(new Int8, new Int8Array(numbers)); });
  test('Int16', () => { testNumberall(new Int16, new Int16Array(numbers)); });
  test('Int32', () => { testNumberall(new Int32, new Int32Array(numbers)); });
  test('Int64', () => { testBigIntall(new Int64, new BigInt64Array(makeBigInts())); });
  test('Uint8', () => { testNumberall(new Uint8, new Uint8Array(numbers)); });
  test('Uint16', () => { testNumberall(new Uint16, new Uint16Array(numbers)); });
  test('Uint32', () => { testNumberall(new Uint32, new Uint32Array(numbers)); });
  test('Uint64', () => { testBigIntall(new Uint64, new BigUint64Array(makeBigInts())); });
  test('Float32', () => { testNumberall(new Float32, new Float32Array(numbers)); });
  test('Float64', () => { testNumberall(new Float64, new Float64Array(numbers)); });
  test('Bool8', () => { testNumberall(new Bool8, new Uint8ClampedArray(makeBooleans())); });
});

describe("Series.all(skipna=false)", () => {
  test('Int8', () => {testNumberallSkipNA(new Int8, new Int8Array(numbers), makeBooleans())});
  test('Int16', () => { testNumberallSkipNA(new Int16, new Int16Array(numbers), makeBooleans()); });
  test('Int32', () => { testNumberallSkipNA(new Int32, new Int32Array(numbers), makeBooleans()); });
  test('Int64',
       () => { testBigIntallSkipNA(new Int64, new BigInt64Array(makeBigInts()), makeBooleans()); });
  test('Uint8', () => { testNumberallSkipNA(new Uint8, new Uint8Array(numbers), makeBooleans()); });
  test('Uint16',
       () => { testNumberallSkipNA(new Uint16, new Uint16Array(numbers), makeBooleans()); });
  test('Uint32',
       () => { testNumberallSkipNA(new Uint32, new Uint32Array(numbers), makeBooleans()); });
  test(
    'Uint64',
    () => { testBigIntallSkipNA(new Uint64, new BigUint64Array(makeBigInts()), makeBooleans()); });
  test('Float32',
       () => { testNumberallSkipNA(new Float32, new Float32Array(numbers), makeBooleans()); });
  test('Float64',
       () => { testNumberallSkipNA(new Float64, new Float64Array(numbers), makeBooleans()); });
  test('Bool8', () => {
    testNumberallSkipNA(new Bool8, new Uint8ClampedArray(makeBooleans()), makeBooleans());
  });
});

describe("Float type Series with NaN => Series.all(skipna=true)", () => {
  test('Float32', () => { testNumberall(new Float32, new Float32Array(float_with_NaN)); });
  test('Float64', () => { testNumberall(new Float64, new Float64Array(float_with_NaN)); });
});

describe("Float type Series with NaN => Series.all(skipna=false)", () => {
  test(
    'Float32',
    () => { testNumberallSkipNA(new Float32, new Float32Array(float_with_NaN), makeBooleans()); });
  test(
    'Float64',
    () => { testNumberallSkipNA(new Float64, new Float64Array(float_with_NaN), makeBooleans()); });
});
