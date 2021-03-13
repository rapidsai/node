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
import {Bool8, Float32, Float64, Int64, Numeric, Uint64} from '@rapidsai/cudf';
import {DeviceBuffer} from '@rapidsai/rmm';
import * as arrow from 'apache-arrow';

import {
  makeTestBigInts,
  makeTestSeries,
  MathematicalUnaryOp,
  testForEachNumericType
} from '../utils';

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength));

const makeTestData = (values?: (number|bigint|null)[]) =>
  makeTestSeries(new arrow.Uint64, makeTestBigInts(values)).lhs;

const toBigInt = (x: bigint|null) => x == null ? null : BigInt.asUintN(64, 0n + x);

describe('Series unaryops (Uint64)', () => {
  const BIG_INT53 = 9223372036854775808n;
  const BIG_INT64 = 18446744073709551615n;
  const runMathOp =
    (op: MathematicalUnaryOp) => [...makeTestData([-3n, 0n, 3n])[op]()].map(toBigInt);
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
    const actual = makeTestData([-3n, 0n, 3n]).not();
    expect([...actual]).toEqual([-3n, 0n, 3n].map((x) => !x));
  });
  test('Series.isNull', () => {
    const actual = makeTestData([null, 3n, 6n]).isNull();
    expect([...actual]).toEqual([true, false, false]);
  });
  test('Series.isValid', () => {
    const actual = makeTestData([null, 3n, 6n]).isValid();
    expect([...actual]).toEqual([false, true, true]);
  });
  test('Series.bit_invert', () => {
    const actual = makeTestData([null, 0n, 3n, 6n]).bit_invert();
    expect([...actual].map(toBigInt)).toEqual([null, ~0n, ~3n, ~6n].map(toBigInt));
  });
  testForEachNumericType(
    'Series.cast %p',
    function testSeriesCast<T extends TypedArray|BigIntArray, R extends Numeric>(
      TypedArrayCtor: TypedArrayConstructor<T>, type: R) {
      const input    = Array.from({length: 16}, () => 16 * (Math.random() - 0.5) | 0);
      const actual   = makeTestData(input).cast(type).data.toArray();
      const expected = new TypedArrayCtor(input.map((x) => {
        if (type instanceof Bool8) { return x | 0 ? 1 : 0; }
        if (type instanceof Int64) { return BigInt.asIntN(64, BigInt(x | 0)); }
        if (type instanceof Uint64) { return BigInt.asUintN(64, BigInt(x | 0)); }
        if (type instanceof Float32) { return Number(BigInt.asUintN(64, BigInt(x | 0))); }
        if (type instanceof Float64) { return Number(BigInt.asUintN(64, BigInt(x | 0))); }
        return x | 0;
      }));
      expect(actual).toEqualTypedArray(expected);
    });
  testForEachNumericType(
    'Series.view %p',
    function testSeriesView<T extends TypedArray|BigIntArray, R extends Numeric>(
      TypedArrayCtor: TypedArrayConstructor<T>, type: R) {
      const input    = Array.from({length: 16}, () => 16 * (Math.random() - 0.5) | 0);
      const actual   = makeTestData(input).view(type).data.toArray();
      const expected = new TypedArrayCtor(new BigUint64Array(input.map(BigInt)).buffer);
      expect(actual).toEqualTypedArray(expected);
    });
});
