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

import {BigIntArray, setDefaultAllocator, TypedArray} from '@nvidia/cuda';
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

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength));

const makeNumbers = (length = 10) => Array.from({length}, (_, i) => Number(i));
const makeBigInts = (length = 10) => Array.from({length}, (_, i) => BigInt(i));
const makeBooleans = (length = 10) => Array.from({length}, (_, i) => Number(i % 2 == 0));

const param_ddof         = [1, 3, 5];
const var_number_results = new Map([[1, 9.166666666666668], [3, 11.785714285714288], [5, 16.5]]);

const var_bool_results = new Map([[1, 0.2777777777777778], [3, 0.35714285714285715], [5, 0.5]]);

function testNumberVar<T extends Numeric, R extends TypedArray|BigIntArray>(
  skipna: boolean, ddof: number, type: T, data: R) {
  expect(Series.new({type, data}).var(skipna, ddof))
    .toEqual((data.includes(<never>NaN) && skipna == false) ? NaN : var_number_results.get(ddof));
}

function testBooleanVar<T extends Numeric, R extends TypedArray|BigIntArray>(
  skipna: boolean, ddof: number, type: T, data: R) {
  expect(Series.new({type, data}).var(skipna, ddof))
    .toEqual((data.includes(<never>NaN) && skipna == false) ? NaN : var_bool_results.get(ddof));
}

param_ddof.forEach(ddof => {
  describe('Series.var (skipna=True)', () => {
    const skipna = true;
    test('Int8', () => { testNumberVar(skipna, ddof, new Int8, new Int8Array(makeNumbers())); });
    test('Int16', () => { testNumberVar(skipna, ddof, new Int16, new Int16Array(makeNumbers())); });
    test('Int32', () => { testNumberVar(skipna, ddof, new Int32, new Int32Array(makeNumbers())); });
    test('Int64',
         () => { testNumberVar(skipna, ddof, new Int64, new BigInt64Array(makeBigInts())); });
    test('Uint8', () => { testNumberVar(skipna, ddof, new Uint8, new Uint8Array(makeNumbers())); });
    test('Uint16',
         () => { testNumberVar(skipna, ddof, new Uint16, new Uint16Array(makeNumbers())); });
    test('Uint32',
         () => { testNumberVar(skipna, ddof, new Uint32, new Uint32Array(makeNumbers())); });
    test('Uint64',
         () => { testNumberVar(skipna, ddof, new Uint64, new BigUint64Array(makeBigInts())); });
    test('Float32',
         () => { testNumberVar(skipna, ddof, new Float32, new Float32Array(makeNumbers())); });
    test('Float64',
         () => { testNumberVar(skipna, ddof, new Float64, new Float64Array(makeNumbers())); });
    test('Bool8',
         () => { testBooleanVar(skipna, ddof, new Bool8, new Uint8ClampedArray(makeBooleans())); });
  });

  describe('Float type Series with NaN => Series.var (skipna=True)', () => {
    const skipna = true;
    test('Float32', () => {
      testNumberVar(
        skipna, ddof, new Float32, new Float32Array([NaN].concat(makeNumbers().concat([NaN]))));
    });
    test('Float64', () => {
      testNumberVar(
        skipna, ddof, new Float64, new Float64Array([NaN].concat(makeNumbers().concat([NaN]))));
    });
  });

  describe('Series.var (skipna=false)', () => {
    const skipna = false;
    test('Int8', () => { testNumberVar(skipna, ddof, new Int8, new Int8Array(makeNumbers())); });
    test('Int16', () => { testNumberVar(skipna, ddof, new Int16, new Int16Array(makeNumbers())); });
    test('Int32', () => { testNumberVar(skipna, ddof, new Int32, new Int32Array(makeNumbers())); });
    test('Int64',
         () => { testNumberVar(skipna, ddof, new Int64, new BigInt64Array(makeBigInts())); });
    test('Uint8', () => { testNumberVar(skipna, ddof, new Uint8, new Uint8Array(makeNumbers())); });
    test('Uint16',
         () => { testNumberVar(skipna, ddof, new Uint16, new Uint16Array(makeNumbers())); });
    test('Uint32',
         () => { testNumberVar(skipna, ddof, new Uint32, new Uint32Array(makeNumbers())); });
    test('Uint64',
         () => { testNumberVar(skipna, ddof, new Uint64, new BigUint64Array(makeBigInts())); });
    test('Float32',
         () => { testNumberVar(skipna, ddof, new Float32, new Float32Array(makeNumbers())); });
    test('Float64',
         () => { testNumberVar(skipna, ddof, new Float64, new Float64Array(makeNumbers())); });
    test('Bool8',
         () => { testBooleanVar(skipna, ddof, new Bool8, new Uint8ClampedArray(makeBooleans())); });
  });

  describe('Float type Series with NaN => Series.var (skipna=false)', () => {
    const skipna = false;
    test('Float32', () => {
      testNumberVar(
        skipna, ddof, new Float32, new Float32Array([NaN].concat(makeNumbers().concat([NaN]))));
    });
    test('Float64', () => {
      testNumberVar(
        skipna, ddof, new Float64, new Float64Array([NaN].concat(makeNumbers().concat([NaN]))));
    });
  });
});
