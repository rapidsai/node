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

import {BigIntArray, setDefaultAllocator, TypedArray, TypedArrayConstructor} from '@rapidsai/cuda';
import {Numeric, Utf8String} from '@rapidsai/cudf';
import {DeviceBuffer} from '@rapidsai/rmm';
import * as arrow from 'apache-arrow';

import {
  clampIntValuesLikeUnaryCast,
  makeTestBigInts,
  makeTestSeries,
  MathematicalUnaryOp,
  testForEachNumericType
} from '../utils';

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength));

const makeTestData = (values?: (number|bigint|null)[]) =>
  makeTestSeries(new arrow.Int64, makeTestBigInts(values)).lhs;

const toBigInt = (x: bigint|null) => x == null ? null : BigInt.asIntN(64, 0n + x);

describe('Series unaryops (Int64)', () => {
  const LIL_INT53 = -9223372036854775808n;
  const runMathOp =
    (op: MathematicalUnaryOp) => [...makeTestData([-3n, 0n, 3n])[op]()].map(toBigInt);
  test('Series.sin', () => { expect(runMathOp('sin')).toEqual([0n, 0n, 0n]); });
  test('Series.cos', () => { expect(runMathOp('cos')).toEqual([0n, 1n, 0n]); });
  test('Series.tan', () => { expect(runMathOp('tan')).toEqual([0n, 0n, 0n]); });
  test('Series.asin', () => { expect(runMathOp('asin')).toEqual([LIL_INT53, 0n, LIL_INT53]); });
  test('Series.acos', () => { expect(runMathOp('acos')).toEqual([LIL_INT53, 1n, LIL_INT53]); });
  test('Series.atan', () => { expect(runMathOp('atan')).toEqual([-1n, 0n, 1n]); });
  test('Series.sinh', () => { expect(runMathOp('sinh')).toEqual([-10n, 0n, 10n]); });
  test('Series.cosh', () => { expect(runMathOp('cosh')).toEqual([10n, 1n, 10n]); });
  test('Series.tanh', () => { expect(runMathOp('tanh')).toEqual([0n, 0n, 0n]); });
  test('Series.asinh', () => { expect(runMathOp('asinh')).toEqual([-1n, 0n, 1n]); });
  test('Series.acosh', () => { expect(runMathOp('acosh')).toEqual([LIL_INT53, LIL_INT53, 1n]); });
  test('Series.atanh', () => { expect(runMathOp('atanh')).toEqual([LIL_INT53, 0n, LIL_INT53]); });
  test('Series.exp', () => { expect(runMathOp('exp')).toEqual([0n, 1n, 20n]); });
  test('Series.log', () => { expect(runMathOp('log')).toEqual([LIL_INT53, LIL_INT53, 1n]); });
  test('Series.sqrt', () => { expect(runMathOp('sqrt')).toEqual([LIL_INT53, 0n, 1n]); });
  test('Series.cbrt', () => { expect(runMathOp('cbrt')).toEqual([-1n, 0n, 1n]); });
  test('Series.ceil', () => { expect(runMathOp('ceil')).toEqual([-3n, 0n, 3n]); });
  test('Series.floor', () => { expect(runMathOp('floor')).toEqual([-3n, 0n, 3n]); });
  test('Series.abs', () => { expect(runMathOp('abs')).toEqual([3n, 0n, 3n]); });
  test('Series.not', () => {
    const actual = makeTestData([-3n, 0n, 3n]).not();
    expect([...actual]).toEqual([-3n, 0n, 3n].map((x) => !x));
  });
  test('Series.isNull', () => {
    const actual = makeTestData([null, 3n, 6n]).isNull();
    expect([...actual]).toEqual([true, false, false]);
  });
  test('Series.isNotNull', () => {
    const actual = makeTestData([null, 3n, 6n]).isNotNull();
    expect([...actual]).toEqual([false, true, true]);
  });
  test('Series.bitInvert', () => {
    const actual = makeTestData([null, 0n, 3n, 6n]).bitInvert();
    expect([...actual].map(toBigInt)).toEqual([null, ~0n, ~3n, ~6n].map(toBigInt));
  });
  test('Series.cast Utf8String', () => {
    const actual = makeTestData([null, 0n, 3n, 6n]);
    expect([...actual.cast(new Utf8String)]).toEqual([null, '0', '3', '6']);
  });
  test('Series.toHexString', () => {
    const actual = makeTestData([1234n, 0n, 27n, null]);
    expect([...actual.toHexString()]).toEqual(['04D2', '00', '1B', null]);
  });
  test('Series.toIpv4String', () => {
    const actual = makeTestData([2080309255n, 2130706433n, null]);
    expect([...actual.toIpv4String()]).toEqual(['123.255.0.7', '127.0.0.1', null]);
  });
  const clampValuesLikeUnaryCast = clampIntValuesLikeUnaryCast(new Int32Array([0]));
  testForEachNumericType(
    'Series.cast %p',
    function testSeriesCast<T extends TypedArray|BigIntArray, R extends Numeric>(
      TypedArrayCtor: TypedArrayConstructor<T>, type: R) {
      const input    = Array.from({length: 16}, () => 16 * (Math.random() - 0.5) | 0);
      const actual   = makeTestData(input).cast(type).data.toArray();
      const expected = new TypedArrayCtor(clampValuesLikeUnaryCast(type, input));
      expect(actual).toEqualTypedArray(expected);
    });
  testForEachNumericType(
    'Series.view %p',
    function testSeriesView<T extends TypedArray|BigIntArray, R extends Numeric>(
      TypedArrayCtor: TypedArrayConstructor<T>, type: R) {
      const input    = Array.from({length: 16}, () => 16 * (Math.random() - 0.5) | 0);
      const actual   = makeTestData(input).view(type).data.toArray();
      const expected = new TypedArrayCtor(new BigInt64Array(input.map(BigInt)).buffer);
      expect(actual).toEqualTypedArray(expected);
    });
});
