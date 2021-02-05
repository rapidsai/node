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

function testNumberSum<T extends Numeric, R extends TypedArray>(type: T, data: R) {
  expect(Series.new({type, data}).sum()).toEqual([...data].reduce((x, y) => x + y));
}

function testBigIntSum<T extends Numeric, R extends BigIntArray>(type: T, data: R) {
  expect(Series.new({type, data}).sum()).toEqual([...data].reduce((x, y) => x + y));
}

describe('Series.sum()', () => {
  test('Int8', () => { testNumberSum(new Int8, new Int8Array(makeNumbers())); });
  test('Int16', () => { testNumberSum(new Int16, new Int16Array(makeNumbers())); });
  test('Int32', () => { testNumberSum(new Int32, new Int32Array(makeNumbers())); });
  test('Int64', () => { testBigIntSum(new Int64, new BigInt64Array(makeBigInts())); });
  test('Uint8', () => { testNumberSum(new Uint8, new Uint8Array(makeNumbers())); });
  test('Uint16', () => { testNumberSum(new Uint16, new Uint16Array(makeNumbers())); });
  test('Uint32', () => { testNumberSum(new Uint32, new Uint32Array(makeNumbers())); });
  test('Uint64', () => { testBigIntSum(new Uint64, new BigUint64Array(makeBigInts())); });
  test('Float32', () => { testNumberSum(new Float32, new Float32Array(makeNumbers())); });
  test('Float64', () => { testNumberSum(new Float64, new Float64Array(makeNumbers())); });
  test('Bool8', () => { testNumberSum(new Bool8, new Uint8ClampedArray(makeBooleans())); });
});
