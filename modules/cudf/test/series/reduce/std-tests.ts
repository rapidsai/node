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
} from '@rapidsai/cudf';
import {DeviceBuffer} from '@rapidsai/rmm';

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength));

const makeNumbers = (length = 10) => Array.from({length}, (_, i) => Number(i));
const makeBigInts = (length = 10) => Array.from({length}, (_, i) => BigInt(i));
const makeBooleans = (length = 10) => Array.from({length}, (_, i) => Number(i % 2 == 0));

const param_ddof = [1, 3, 5];
const std_number_results =
  new Map([[1, 3.0276503540974917], [3, 3.4330328116279762], [5, 4.06201920231798]]);

const std_bool_results =
  new Map([[1, 0.5270462766947299], [3, 0.5976143046671968], [5, 0.7071067811865476]]);

function testNumberStd<T extends Numeric, R extends TypedArray|BigIntArray>(
  skipNulls: boolean, ddof: number, type: T, data: R) {
  expect(Series.new({type, data}).std(skipNulls, ddof))
    .toEqual((data.includes(<never>NaN) && skipNulls == false) ? NaN
                                                               : std_number_results.get(ddof));
}

function testBooleanStd<T extends Numeric, R extends TypedArray|BigIntArray>(
  skipNulls: boolean, ddof: number, type: T, data: R) {
  expect(Series.new({type, data}).std(skipNulls, ddof))
    .toEqual((data.includes(<never>NaN) && skipNulls == false) ? NaN : std_bool_results.get(ddof));
}

param_ddof.forEach(ddof => {
  describe('Series.std (skipNulls=True)', () => {
    const skipNulls = true;
    test('Int8', () => { testNumberStd(skipNulls, ddof, new Int8, new Int8Array(makeNumbers())); });
    test('Int16',
         () => { testNumberStd(skipNulls, ddof, new Int16, new Int16Array(makeNumbers())); });
    test('Int32',
         () => { testNumberStd(skipNulls, ddof, new Int32, new Int32Array(makeNumbers())); });
    test('Int64',
         () => { testNumberStd(skipNulls, ddof, new Int64, new BigInt64Array(makeBigInts())); });
    test('Uint8',
         () => { testNumberStd(skipNulls, ddof, new Uint8, new Uint8Array(makeNumbers())); });
    test('Uint16',
         () => { testNumberStd(skipNulls, ddof, new Uint16, new Uint16Array(makeNumbers())); });
    test('Uint32',
         () => { testNumberStd(skipNulls, ddof, new Uint32, new Uint32Array(makeNumbers())); });
    test('Uint64',
         () => { testNumberStd(skipNulls, ddof, new Uint64, new BigUint64Array(makeBigInts())); });
    test('Float32',
         () => { testNumberStd(skipNulls, ddof, new Float32, new Float32Array(makeNumbers())); });
    test('Float64',
         () => { testNumberStd(skipNulls, ddof, new Float64, new Float64Array(makeNumbers())); });
    test(
      'Bool8',
      () => { testBooleanStd(skipNulls, ddof, new Bool8, new Uint8ClampedArray(makeBooleans())); });
  });

  describe('Float type Series with NaN => Series.std (skipNulls=True)', () => {
    const skipNulls = true;
    test('Float32', () => {
      testNumberStd(
        skipNulls, ddof, new Float32, new Float32Array([NaN].concat(makeNumbers().concat([NaN]))));
    });
    test('Float64', () => {
      testNumberStd(
        skipNulls, ddof, new Float64, new Float64Array([NaN].concat(makeNumbers().concat([NaN]))));
    });
  });

  describe('Series.std (skipNulls=false)', () => {
    const skipNulls = false;
    test('Int8', () => { testNumberStd(skipNulls, ddof, new Int8, new Int8Array(makeNumbers())); });
    test('Int16',
         () => { testNumberStd(skipNulls, ddof, new Int16, new Int16Array(makeNumbers())); });
    test('Int32',
         () => { testNumberStd(skipNulls, ddof, new Int32, new Int32Array(makeNumbers())); });
    test('Int64',
         () => { testNumberStd(skipNulls, ddof, new Int64, new BigInt64Array(makeBigInts())); });
    test('Uint8',
         () => { testNumberStd(skipNulls, ddof, new Uint8, new Uint8Array(makeNumbers())); });
    test('Uint16',
         () => { testNumberStd(skipNulls, ddof, new Uint16, new Uint16Array(makeNumbers())); });
    test('Uint32',
         () => { testNumberStd(skipNulls, ddof, new Uint32, new Uint32Array(makeNumbers())); });
    test('Uint64',
         () => { testNumberStd(skipNulls, ddof, new Uint64, new BigUint64Array(makeBigInts())); });
    test('Float32',
         () => { testNumberStd(skipNulls, ddof, new Float32, new Float32Array(makeNumbers())); });
    test('Float64',
         () => { testNumberStd(skipNulls, ddof, new Float64, new Float64Array(makeNumbers())); });
    test(
      'Bool8',
      () => { testBooleanStd(skipNulls, ddof, new Bool8, new Uint8ClampedArray(makeBooleans())); });
  });

  describe('Float type Series with NaN => Series.std (skipNulls=false)', () => {
    const skipNulls = false;
    test('Float32', () => {
      testNumberStd(
        skipNulls, ddof, new Float32, new Float32Array([NaN].concat(makeNumbers().concat([NaN]))));
    });
    test('Float64', () => {
      testNumberStd(
        skipNulls, ddof, new Float64, new Float64Array([NaN].concat(makeNumbers().concat([NaN]))));
    });
  });
});
