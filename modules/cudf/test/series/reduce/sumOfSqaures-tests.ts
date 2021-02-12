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

const makeNumbers = (length = 10) => Array.from({length}, (_, i) => Number(i));

const makeBigInts = (length = 10) => Array.from({length}, (_, i) => BigInt(i));

const makeBooleans = (length = 10) => Array.from({length}, (_, i) => Number(i % 2 == 0));

const float_with_NaN = Array.from([NaN, 1, 2, 3, 4, 5, 6, 7, 8, NaN]);

function test_values<T extends Numeric>(lhs: number|bigint|undefined, rhs: number, type: T) {
  if (["Float32", "Float64", "Bool"].includes(type.toString())) {
    expect(lhs).toEqual(rhs);
  } else {
    expect(lhs).toEqual(BigInt(rhs));
  }
}

function testNumberSumOfSquares<T extends Numeric, R extends TypedArray>(type: T, data: R) {
  const result = [...data].reduce((x, y) => {
    if (isNaN(y)) { y = 0; }
    if (isNaN(x)) { x = 0; }
    return x + Math.pow(y, 2);
  });
  test_values(Series.new({type, data}).sumOfSquares(), result, type);
}

function testNumberSumOfSquaresSkipNA<T extends Numeric, R extends TypedArray>(
  type: T, data: R, mask_array: Array<number>) {
  const mask = new Uint8Buffer(BoolVector.from(mask_array).values);
  if (!data.includes(NaN)) {
    const result = [...data].reduce((x, y, i) => {
      if (mask_array[i] == 1) { return x + Math.pow(y, 2); }
      return x;
    });
    // skipna=false
    test_values(Series.new({type, data, nullMask: mask}).sumOfSquares(false), result, type);
  } else {
    // skipna=false
    expect(Series.new({type, data, nullMask: mask}).sumOfSquares(false)).toEqual(NaN);
  }
}

function testBigIntSumOfSquares<T extends Numeric, R extends BigIntArray>(type: T, data: R) {
  expect(Series.new({type, data}).sumOfSquares()).toEqual([...data].reduce((x, y) => x + (y * y)));
}

function testBigIntSumOfSquaresSkipNA<T extends Numeric, R extends BigIntArray>(
  type: T, data: R, mask_array: Array<number>) {
  const mask = new Uint8Buffer(BoolVector.from(mask_array).values);
  // skipna=false
  expect(Series.new({type, data, nullMask: mask}).sumOfSquares(false)).toEqual([
    ...data
  ].reduce((x, y, i) => {
    if (mask_array[i] == 1) { return x + (y * y); }
    return x;
  }));
}

describe('Series.sumOfSquares(skipna=true)', () => {
  test('Int8', () => { testNumberSumOfSquares(new Int8, new Int8Array(makeNumbers())); });
  test('Int16', () => { testNumberSumOfSquares(new Int16, new Int16Array(makeNumbers())); });
  test('Int32', () => { testNumberSumOfSquares(new Int32, new Int32Array(makeNumbers())); });
  test('Int64', () => { testBigIntSumOfSquares(new Int64, new BigInt64Array(makeBigInts())); });
  test('Uint8', () => { testNumberSumOfSquares(new Uint8, new Uint8Array(makeNumbers())); });
  test('Uint16', () => { testNumberSumOfSquares(new Uint16, new Uint16Array(makeNumbers())); });
  test('Uint32', () => { testNumberSumOfSquares(new Uint32, new Uint32Array(makeNumbers())); });
  test('Uint64', () => { testBigIntSumOfSquares(new Uint64, new BigUint64Array(makeBigInts())); });
  test('Float32', () => { testNumberSumOfSquares(new Float32, new Float32Array(makeNumbers())); });
  test('Float64', () => { testNumberSumOfSquares(new Float64, new Float64Array(makeNumbers())); });
  test('Bool8',
       () => { testNumberSumOfSquares(new Bool8, new Uint8ClampedArray(makeBooleans())); });
});

describe("Series.sumOfSquares(skipna=false)", () => {
  test(
    'Int8',
    () => {testNumberSumOfSquaresSkipNA(new Int8, new Int8Array(makeNumbers()), makeBooleans())});
  test('Int16', () => {
    testNumberSumOfSquaresSkipNA(new Int16, new Int16Array(makeNumbers()), makeBooleans());
  });
  test('Int32', () => {
    testNumberSumOfSquaresSkipNA(new Int32, new Int32Array(makeNumbers()), makeBooleans());
  });
  test('Int64', () => {
    testBigIntSumOfSquaresSkipNA(new Int64, new BigInt64Array(makeBigInts()), makeBooleans());
  });
  test('Uint8', () => {
    testNumberSumOfSquaresSkipNA(new Uint8, new Uint8Array(makeNumbers()), makeBooleans());
  });
  test('Uint16', () => {
    testNumberSumOfSquaresSkipNA(new Uint16, new Uint16Array(makeNumbers()), makeBooleans());
  });
  test('Uint32', () => {
    testNumberSumOfSquaresSkipNA(new Uint32, new Uint32Array(makeNumbers()), makeBooleans());
  });
  test('Uint64', () => {
    testBigIntSumOfSquaresSkipNA(new Uint64, new BigUint64Array(makeBigInts()), makeBooleans());
  });
  test('Float32', () => {
    testNumberSumOfSquaresSkipNA(new Float32, new Float32Array(makeNumbers()), makeBooleans());
  });
  test('Float64', () => {
    testNumberSumOfSquaresSkipNA(new Float64, new Float64Array(makeNumbers()), makeBooleans());
  });
  test('Bool8', () => {
    testNumberSumOfSquaresSkipNA(new Bool8, new Uint8ClampedArray(makeBooleans()), makeBooleans());
  });
});

describe("Float type Series with NaN => Series.sumOfSquares(skipna=true)", () => {
  test('Float32', () => { testNumberSumOfSquares(new Float32, new Float32Array(float_with_NaN)); });
  test('Float64', () => { testNumberSumOfSquares(new Float64, new Float64Array(float_with_NaN)); });
});

describe("Float type Series with NaN => Series.sumOfSquares(skipna=false)", () => {
  test('Float32', () => {
    testNumberSumOfSquaresSkipNA(new Float32, new Float32Array(float_with_NaN), makeBooleans());
  });
  test('Float64', () => {
    testNumberSumOfSquaresSkipNA(new Float64, new Float64Array(float_with_NaN), makeBooleans());
  });
});
