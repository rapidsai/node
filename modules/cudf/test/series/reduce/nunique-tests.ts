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
import {BoolVector} from 'apache-arrow';

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength));

const numbers = Array.from([null, 0, 1, 1, null, 2, 3, 3, 4, 4]) as number[];

const makeBigInts = (length = 10) => Array.from({length}, (_, i) => BigInt(Math.floor(i / 3)));

const makeBooleans = (length = 10) => Array.from({length}, (_, i) => Number(i % 2 == 0));

const float_with_NaN = Array.from([NaN, 1, 2, 3, 4, 3, 7, 7, 2, NaN]);

function jsNunique(values: any) {
  return values.filter((x: any, i: any, self: string|any[]) => self.indexOf(x) == i).length;
}

function testNumberNunique<T extends Numeric, R extends TypedArray>(type: T, data: R) {
  expect(Series.new({type, data}).nunique()).toEqual(jsNunique(data));
}

function testNumberNuniqueSkipNA<T extends Numeric, R extends TypedArray>(
  type: T, data: R, mask_array: Array<number>) {
  const mask = new Uint8Buffer(BoolVector.from(mask_array).values);
  // skipna=false
  expect(Series.new({type, data, nullMask: mask}).nunique(false))
    .toEqual(jsNunique(data.filter((_: any, i: number) => mask_array[i] == 1 || data[i] !== null)));
}

function testBigIntNunique<T extends Numeric, R extends BigIntArray>(type: T, data: R) {
  expect(Series.new({type, data}).nunique()).toEqual(jsNunique(data));
}

function testBigIntNuniqueSkipNA<T extends Numeric, R extends BigIntArray>(
  type: T, data: R, mask_array: Array<number>) {
  const mask = new Uint8Buffer(BoolVector.from(mask_array).values);
  // skipna=false
  expect(Series.new({type, data, nullMask: mask}).nunique(false))
    .toEqual(jsNunique(data.filter((_: any, i: number) => mask_array[i] == 1 || data[i] !== null)));
}

describe('Series.nunique(skipna=true)', () => {
  test('Int8', () => { testNumberNunique(new Int8, new Int8Array(numbers)); });
  test('Int16', () => { testNumberNunique(new Int16, new Int16Array(numbers)); });
  test('Int32', () => { testNumberNunique(new Int32, new Int32Array(numbers)); });
  test('Int64', () => { testBigIntNunique(new Int64, new BigInt64Array(makeBigInts())); });
  test('Uint8', () => { testNumberNunique(new Uint8, new Uint8Array(numbers)); });
  test('Uint16', () => { testNumberNunique(new Uint16, new Uint16Array(numbers)); });
  test('Uint32', () => { testNumberNunique(new Uint32, new Uint32Array(numbers)); });
  test('Uint64', () => { testBigIntNunique(new Uint64, new BigUint64Array(makeBigInts())); });
  test('Float32', () => { testNumberNunique(new Float32, new Float32Array(numbers)); });
  test('Float64', () => { testNumberNunique(new Float64, new Float64Array(numbers)); });
  test('Bool8', () => { testNumberNunique(new Bool8, new Uint8ClampedArray(makeBooleans())); });
});

describe('Series.nunique(skipna=false)', () => {
  test('Int8', () => {testNumberNuniqueSkipNA(new Int8, new Int8Array(numbers), makeBooleans())});
  test('Int16',
       () => { testNumberNuniqueSkipNA(new Int16, new Int16Array(numbers), makeBooleans()); });
  test('Int32',
       () => { testNumberNuniqueSkipNA(new Int32, new Int32Array(numbers), makeBooleans()); });
  test('Int64', () => {
    testBigIntNuniqueSkipNA(new Int64, new BigInt64Array(makeBigInts()), makeBooleans());
  });
  test('Uint8',
       () => { testNumberNuniqueSkipNA(new Uint8, new Uint8Array(numbers), makeBooleans()); });
  test('Uint16',
       () => { testNumberNuniqueSkipNA(new Uint16, new Uint16Array(numbers), makeBooleans()); });
  test('Uint32',
       () => { testNumberNuniqueSkipNA(new Uint32, new Uint32Array(numbers), makeBooleans()); });
  test('Uint64', () => {
    testBigIntNuniqueSkipNA(new Uint64, new BigUint64Array(makeBigInts()), makeBooleans());
  });
  test('Float32',
       () => { testNumberNuniqueSkipNA(new Float32, new Float32Array(numbers), makeBooleans()); });
  test('Float64',
       () => { testNumberNuniqueSkipNA(new Float64, new Float64Array(numbers), makeBooleans()); });
  test('Bool8', () => {
    testNumberNuniqueSkipNA(new Bool8, new Uint8ClampedArray(makeBooleans()), makeBooleans());
  });
});

describe('Float type Series with NaN => Series.nunique(skipna=true)', () => {
  test('Float32', () => { testNumberNunique(new Float32, new Float32Array(float_with_NaN)); });
  test('Float64', () => { testNumberNunique(new Float64, new Float64Array(float_with_NaN)); });
});

describe('Float type Series with NaN => Series.nunique(skipna=false)', () => {
  test('Float32', () => {
    testNumberNuniqueSkipNA(new Float32, new Float32Array(float_with_NaN), makeBooleans());
  });
  test('Float64', () => {
    testNumberNuniqueSkipNA(new Float64, new Float64Array(float_with_NaN), makeBooleans());
  });
});
