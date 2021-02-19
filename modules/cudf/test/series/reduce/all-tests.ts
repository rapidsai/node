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

function jsAll(values: any) { return values.every(x => x != false); }

function testNumberAll<T extends Numeric, R extends TypedArray>(type: T, data: R) {
  expect(Boolean(Series.new({type, data}).all())).toEqual(jsAll(data));
}

function testNumberAllSkipNA<T extends Numeric, R extends TypedArray>(
  type: T, data: R, mask_array: Array<number>) {
  const mask = new Uint8Buffer(BoolVector.from(mask_array).values);
  // skipna=false
  expect(Boolean(Series.new({type, data, nullMask: mask}).all(false)))
    .toEqual(jsAll(data.filter((_, i) => mask_array[i] == 1)));
}

function testBigIntAll<T extends Numeric, R extends BigIntArray>(type: T, data: R) {
  expect(Boolean(Series.new({type, data}).all())).toEqual(jsAll(data));
}

function testBigIntAllSkipNA<T extends Numeric, R extends BigIntArray>(
  type: T, data: R, mask_array: Array<number>) {
  const mask = new Uint8Buffer(BoolVector.from(mask_array).values);
  // skipna=false
  expect(Boolean(Series.new({type, data, nullMask: mask}).all(false)))
    .toEqual(jsAll(data.filter((_, i) => mask_array[i] == 1)));
}

describe('Series.all(skipna=true)', () => {
  test('Int8', () => { testNumberAll(new Int8, new Int8Array(numbers)); });
  test('Int16', () => { testNumberAll(new Int16, new Int16Array(numbers)); });
  test('Int32', () => { testNumberAll(new Int32, new Int32Array(numbers)); });
  test('Int64', () => { testBigIntAll(new Int64, new BigInt64Array(makeBigInts())); });
  test('Uint8', () => { testNumberAll(new Uint8, new Uint8Array(numbers)); });
  test('Uint16', () => { testNumberAll(new Uint16, new Uint16Array(numbers)); });
  test('Uint32', () => { testNumberAll(new Uint32, new Uint32Array(numbers)); });
  test('Uint64', () => { testBigIntAll(new Uint64, new BigUint64Array(makeBigInts())); });
  test('Float32', () => { testNumberAll(new Float32, new Float32Array(numbers)); });
  test('Float64', () => { testNumberAll(new Float64, new Float64Array(numbers)); });
  test('Bool8', () => { testNumberAll(new Bool8, new Uint8ClampedArray(makeBooleans())); });
});

describe("Series.all(skipna=false)", () => {
  test('Int8', () => {testNumberAllSkipNA(new Int8, new Int8Array(numbers), makeBooleans())});
  test('Int16', () => { testNumberAllSkipNA(new Int16, new Int16Array(numbers), makeBooleans()); });
  test('Int32', () => { testNumberAllSkipNA(new Int32, new Int32Array(numbers), makeBooleans()); });
  test('Int64',
       () => { testBigIntAllSkipNA(new Int64, new BigInt64Array(makeBigInts()), makeBooleans()); });
  test('Uint8', () => { testNumberAllSkipNA(new Uint8, new Uint8Array(numbers), makeBooleans()); });
  test('Uint16',
       () => { testNumberAllSkipNA(new Uint16, new Uint16Array(numbers), makeBooleans()); });
  test('Uint32',
       () => { testNumberAllSkipNA(new Uint32, new Uint32Array(numbers), makeBooleans()); });
  test(
    'Uint64',
    () => { testBigIntAllSkipNA(new Uint64, new BigUint64Array(makeBigInts()), makeBooleans()); });
  test('Float32',
       () => { testNumberAllSkipNA(new Float32, new Float32Array(numbers), makeBooleans()); });
  test('Float64',
       () => { testNumberAllSkipNA(new Float64, new Float64Array(numbers), makeBooleans()); });
  test('Bool8', () => {
    testNumberAllSkipNA(new Bool8, new Uint8ClampedArray(makeBooleans()), makeBooleans());
  });
});

describe("Float type Series with NaN => Series.all(skipna=true)", () => {
  test('Float32', () => { testNumberAll(new Float32, new Float32Array(float_with_NaN)); });
  test('Float64', () => { testNumberAll(new Float64, new Float64Array(float_with_NaN)); });
});

describe("Float type Series with NaN => Series.all(skipna=false)", () => {
  test(
    'Float32',
    () => { testNumberAllSkipNA(new Float32, new Float32Array(float_with_NaN), makeBooleans()); });
  test(
    'Float64',
    () => { testNumberAllSkipNA(new Float64, new Float64Array(float_with_NaN), makeBooleans()); });
});
