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

import {
  BigIntArray,
  MemoryViewConstructor,
  setDefaultAllocator,
  TypedArray,
  TypedArrayConstructor
} from '@nvidia/cuda';
import {Numeric} from '@nvidia/cudf';
import {DeviceBuffer} from '@nvidia/rmm';
import * as arrow from 'apache-arrow';
import {zip} from 'ix/iterable';

import {
  clampFloatValuesLikeUnaryCast,
  makeTestNumbers,
  makeTestSeries,
  MathematicalUnaryOp,
  mathematicalUnaryOps,
  testForEachNumericType
} from '../utils';

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength));

const makeTestData = (values?: (number|null)[]) =>
  makeTestSeries(new arrow.Float64, makeTestNumbers(values)).lhs;

describe('Series unaryops (Float64)', () => {
  const testMathematicalOp = <P extends MathematicalUnaryOp>(unaryMathOp: P) => {
    test(`Series.${unaryMathOp}`, () => {
      const values   = [-2.5, 0, 2.5];
      const actual   = [...makeTestData(values)[unaryMathOp]()];
      const expected = new Float64Array(values).map(x => Math[unaryMathOp](x));
      for (const [x, y] of zip(actual, expected)) {
        (isNaN(x) && isNaN(y)) ? expect(x).toBeNaN() : expect(x).toBeCloseTo(y);
      }
    });
  };
  for (const op of mathematicalUnaryOps) { testMathematicalOp(op); }
  test('Series.not', () => {
    const actual = makeTestData([-2.5, 0, 2.5]).not();
    expect([...actual]).toEqual([-2.5, 0, 2.5].map((x) => !x));
  });
  test('Series.isNull', () => {
    const actual = makeTestData([null, 2.5, 5]).isNull();
    expect([...actual]).toEqual([true, false, false]);
  });
  test('Series.isValid', () => {
    const actual = makeTestData([null, 2.5, 5]).isValid();
    expect([...actual]).toEqual([false, true, true]);
  });
  test('Series.isNaN', () => {
    const actual = makeTestData([NaN, 2.5, 5]).isNaN();
    expect([...actual]).toEqual([true, false, false]);
  });
  test('Series.isNotNaN', () => {
    const actual = makeTestData([NaN, 2.5, 5]).isNotNaN();
    expect([...actual]).toEqual([false, true, true]);
  });
  test('Series.rint', () => {
    const actual = makeTestData([NaN, 2.5, 5]).rint();
    expect([...actual]).toEqual([NaN, 2, 5]);
  });
  testForEachNumericType(
    'Series.cast %p',
    function<T extends TypedArray|BigIntArray, R extends Numeric>(
      TypedArrayCtor: TypedArrayConstructor<T>, MemoryViewCtor: MemoryViewConstructor<T>, type: R) {
      const input    = Array.from({length: 16}, () => 16 * (Math.random() - 0.5));
      const actual   = new MemoryViewCtor(makeTestData(input).cast(type).data).toArray();
      const expected = new TypedArrayCtor(clampFloatValuesLikeUnaryCast(type, input));
      expect(actual.subarray(0, expected.length)).toEqualTypedArray(expected);
    });
  testForEachNumericType(
    'Series.view %p',
    function<T extends TypedArray|BigIntArray, R extends Numeric>(
      TypedArrayCtor: TypedArrayConstructor<T>, MemoryViewCtor: MemoryViewConstructor<T>, type: R) {
      const input    = Array.from({length: 16}, () => 16 * (Math.random() - 0.5));
      const actual   = new MemoryViewCtor(makeTestData(input).view(type).data).toArray();
      const expected = new TypedArrayCtor(new Float64Array(input).buffer);
      expect(actual.subarray(0, expected.length)).toEqualTypedArray(expected);
    });
});
