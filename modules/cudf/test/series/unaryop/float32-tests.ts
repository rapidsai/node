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

import {BigIntArray, setDefaultAllocator, TypedArray, TypedArrayConstructor} from '@nvidia/cuda';
import {Numeric} from '@rapidsai/cudf';
import {DeviceBuffer} from '@rapidsai/rmm';
import * as arrow from 'apache-arrow';

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
  makeTestSeries(new arrow.Float32, makeTestNumbers(values)).lhs;

describe('Series unaryops (Float32)', () => {
  const testMathematicalOp = <P extends MathematicalUnaryOp>(unaryMathOp: P) => {
    test(`Series.${unaryMathOp}`, () => {
      const values   = [-2.5, 0, 2.5];
      const actual   = makeTestData(values)[unaryMathOp]().data.toArray();
      const expected = new Float32Array(values).map(x => Math[unaryMathOp](x));
      expect(actual).toEqualTypedArray(expected);
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
    function testSeriesCast<T extends TypedArray|BigIntArray, R extends Numeric>(
      TypedArrayCtor: TypedArrayConstructor<T>, type: R) {
      const input    = Array.from({length: 16}, () => 16 * (Math.random() - 0.5));
      const actual   = makeTestData(input).cast(type).data.toArray();
      const expected = new TypedArrayCtor(clampFloatValuesLikeUnaryCast(type, input));
      expect(actual).toEqualTypedArray(expected);
    });
  testForEachNumericType(
    'Series.view %p',
    function testSeriesView<T extends TypedArray|BigIntArray, R extends Numeric>(
      TypedArrayCtor: TypedArrayConstructor<T>, type: R) {
      const input    = Array.from({length: 16}, () => 16 * (Math.random() - 0.5));
      const actual   = makeTestData(input).view(type).data.toArray();
      const expected = new TypedArrayCtor(new Float32Array(input).buffer);
      expect(actual).toEqualTypedArray(expected);
    });
});
