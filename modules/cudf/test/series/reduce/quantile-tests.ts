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

import {BigIntArray, setDefaultAllocator, TypedArray} from '@rapidsai/cuda';
import {
  Bool8,
  Float32,
  Float64,
  Int16,
  Int32,
  Int64,
  Int8,
  Interpolation,
  Numeric,
  Series,
  Uint16,
  Uint32,
  Uint64,
  Uint8
} from '@rapidsai/cudf';
import {DeviceBuffer} from '@rapidsai/rmm';

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength));

const makeNumbers = (length = 10) => Array.from({length}, (_, i) => Number(i));
const makeBigInts = (length = 10) => Array.from({length}, (_, i) => BigInt(i));
const makeBooleans = (length = 10) => Array.from({length}, (_, i) => Number(i % 2 == 0));

const param_q = [0.3, 0.7];
const param_interpolation =
  ['linear', 'lower', 'higher', 'midpoint', 'nearest'] as (keyof typeof Interpolation)[];
const quantile_number_results =
  new Map([[0.3, [2.6999999999999997, 2, 3, 2.5, 3]], [0.7, [6.3, 6, 7, 6.5, 6]]]);

const quantile_bool_results = new Map([[0.3, [0, 0, 0, 0, 0]], [0.7, [1, 1, 1, 1, 1]]]);

function testNumberQuantile<T extends Numeric, R extends TypedArray|BigIntArray>(
  q: number, type: T, data: R) {
  param_interpolation.forEach((interop, idx) => {
    expect(Series.new({type, data}).quantile(q, interop))
      .toEqual(quantile_number_results.get(q)?.[idx]);
  });
}

function testBooleanQuantile<T extends Numeric, R extends TypedArray|BigIntArray>(
  q: number, type: T, data: R) {
  param_interpolation.forEach((interop, idx) => {
    expect(Series.new({type, data}).quantile(q, interop))
      .toEqual(quantile_bool_results.get(q)?.[idx]);
  });
}

param_q.forEach(q => {
  describe('Series.quantile', () => {
    test('Int8', () => { testNumberQuantile(q, new Int8, new Int8Array(makeNumbers())); });
    test('Int16', () => { testNumberQuantile(q, new Int16, new Int16Array(makeNumbers())); });
    test('Int32', () => { testNumberQuantile(q, new Int32, new Int32Array(makeNumbers())); });
    test('Int64', () => { testNumberQuantile(q, new Int64, new BigInt64Array(makeBigInts())); });
    test('Uint8', () => { testNumberQuantile(q, new Uint8, new Uint8Array(makeNumbers())); });
    test('Uint16', () => { testNumberQuantile(q, new Uint16, new Uint16Array(makeNumbers())); });
    test('Uint32', () => { testNumberQuantile(q, new Uint32, new Uint32Array(makeNumbers())); });
    test('Uint64', () => { testNumberQuantile(q, new Uint64, new BigUint64Array(makeBigInts())); });
    test('Float32', () => { testNumberQuantile(q, new Float32, new Float32Array(makeNumbers())); });
    test('Float64', () => { testNumberQuantile(q, new Float64, new Float64Array(makeNumbers())); });
    test('Bool8',
         () => { testBooleanQuantile(q, new Bool8, new Uint8ClampedArray(makeBooleans())); });
  });

  describe('Float type Series with NaN => Series.quantile', () => {
    test('Float32', () => {
      testNumberQuantile(
        q, new Float32, new Float32Array([NaN].concat(makeNumbers().concat([NaN]))));
    });
    test('Float64', () => {
      testNumberQuantile(
        q, new Float64, new Float64Array([NaN].concat(makeNumbers().concat([NaN]))));
    });
  });
});
