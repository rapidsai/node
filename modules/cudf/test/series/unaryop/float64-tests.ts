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

import {setDefaultAllocator} from '@nvidia/cuda';
import {DeviceBuffer} from '@nvidia/rmm';
import * as arrow from 'apache-arrow';
import {zip} from 'ix/iterable';

import {makeTestNumbers, makeTestSeries, MathematicalUnaryOp, mathematicalUnaryOps} from '../utils';

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength));

const makeTestData = (values?: (number|null)[]) =>
  makeTestSeries(new arrow.Float64, makeTestNumbers(values));

describe('Series unaryops (Float64)', () => {
  const testMathematicalOp = <P extends MathematicalUnaryOp>(unaryMathOp: P) => {
    test(`Series.${unaryMathOp}`, () => {
      const values   = [-2.5, 0, 2.5];
      const actual   = [...makeTestData(values).lhs[unaryMathOp]()];
      const expected = new Float64Array(values).map(x => Math[unaryMathOp](x));
      for (const [x, y] of zip(actual, expected)) {
        (isNaN(x) && isNaN(y)) ? expect(x).toBeNaN() : expect(x).toBeCloseTo(y);
      }
    });
  };
  for (const op of mathematicalUnaryOps) { testMathematicalOp(op); }
  test('Series.not', () => {
    const {lhs} = makeTestData([-2.5, 0, 2.5]);
    expect([...lhs.not()]).toEqual([-2.5, 0, 2.5].map((x) => !x));
  });
  test('Series.isNull', () => {
    const {lhs} = makeTestData([null, 2.5, 5]);
    expect([...lhs.isNull()]).toEqual([true, false, false]);
  });
  test('Series.isValid', () => {
    const {lhs} = makeTestData([null, 2.5, 5]);
    expect([...lhs.isValid()]).toEqual([false, true, true]);
  });
  test('Series.isNaN', () => {
    const {lhs} = makeTestData([NaN, 2.5, 5]);
    expect([...lhs.isNaN()]).toEqual([true, false, false]);
  });
  test('Series.isNotNaN', () => {
    const {lhs} = makeTestData([NaN, 2.5, 5]);
    expect([...lhs.isNotNaN()]).toEqual([false, true, true]);
  });
  test('Series.rint', () => {
    const {lhs} = makeTestData([NaN, 2.5, 5]);
    expect([...lhs.rint()]).toEqual([NaN, 2, 5]);
  });
});
