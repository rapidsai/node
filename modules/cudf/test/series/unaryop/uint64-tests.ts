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

import {makeTestBigInts, makeTestSeries, MathematicalUnaryOp} from '../utils';

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength));

const makeTestData = (values?: (number|bigint|null)[]) =>
  makeTestSeries(new arrow.Uint64, makeTestBigInts(values));

const toBigInt = (x: bigint|null) => x == null ? null : BigInt.asUintN(64, 0n + x);

describe('Series unaryops (Uint64)', () => {
  const BIG_INT53 = 9223372036854775808n;
  const BIG_INT64 = 18446744073709551615n;
  const runMathOp =
    (op: MathematicalUnaryOp) => [...makeTestData([-3n, 0n, 3n]).lhs[op]()].map(toBigInt);
  test('Series.sin', () => { expect(runMathOp('sin')).toEqual([0n, 0n, 0n]); });
  test('Series.cos', () => { expect(runMathOp('cos')).toEqual([0n, 1n, 0n]); });
  test('Series.tan', () => { expect(runMathOp('tan')).toEqual([0n, 0n, 0n]); });
  test('Series.asin', () => { expect(runMathOp('asin')).toEqual([BIG_INT53, 0n, BIG_INT53]); });
  test('Series.acos', () => { expect(runMathOp('acos')).toEqual([BIG_INT53, 1n, BIG_INT53]); });
  test('Series.atan', () => { expect(runMathOp('atan')).toEqual([1n, 0n, 1n]); });
  test('Series.sinh', () => { expect(runMathOp('sinh')).toEqual([BIG_INT64, 0n, 10n]); });
  test('Series.cosh', () => { expect(runMathOp('cosh')).toEqual([BIG_INT64, 1n, 10n]); });
  test('Series.tanh', () => { expect(runMathOp('tanh')).toEqual([1n, 0n, 0n]); });
  test('Series.asinh', () => { expect(runMathOp('asinh')).toEqual([45n, 0n, 1n]); });
  test('Series.acosh', () => { expect(runMathOp('acosh')).toEqual([45n, BIG_INT53, 1n]); });
  test('Series.atanh', () => { expect(runMathOp('atanh')).toEqual([BIG_INT53, 0n, BIG_INT53]); });
  test('Series.exp', () => { expect(runMathOp('exp')).toEqual([BIG_INT64, 1n, 20n]); });
  test('Series.log', () => { expect(runMathOp('log')).toEqual([44n, 0n, 1n]); });
  test('Series.sqrt', () => { expect(runMathOp('sqrt')).toEqual([4294967296n, 0n, 1n]); });
  test('Series.cbrt', () => { expect(runMathOp('cbrt')).toEqual([2642245n, 0n, 1n]); });
  test('Series.ceil', () => { expect(runMathOp('ceil')).toEqual([BIG_INT64, 0n, 3n]); });
  test('Series.floor', () => { expect(runMathOp('floor')).toEqual([BIG_INT64, 0n, 3n]); });
  test('Series.abs', () => { expect(runMathOp('abs')).toEqual([18446744073709551613n, 0n, 3n]); });
  test('Series.not', () => {
    const {lhs} = makeTestData([-3n, 0n, 3n]);
    expect([...lhs.not()]).toEqual([-3n, 0n, 3n].map((x) => !x));
  });
  test('Series.isNull', () => {
    const {lhs} = makeTestData([null, 3n, 6n]);
    expect([...lhs.isNull()]).toEqual([true, false, false]);
  });
  test('Series.isValid', () => {
    const {lhs} = makeTestData([null, 3n, 6n]);
    expect([...lhs.isValid()]).toEqual([false, true, true]);
  });
  test('Series.bit_invert', () => {
    const {lhs} = makeTestData([null, 0n, 3n, 6n]);
    expect([...lhs.bit_invert()].map(toBigInt)).toEqual([null, ~0n, ~3n, ~6n].map(toBigInt));
  });
});
