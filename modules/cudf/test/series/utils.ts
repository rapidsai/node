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

import {
  BigIntArray,
  TypedArray,
  TypedArrayConstructor,
} from '@rapidsai/cuda';
import {
  Bool8,
  DataType,
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
import * as arrow from 'apache-arrow';

export function toBigInt(value: any) { return BigInt(value == null ? 0n : value); }

export function makeTestNumbers(values: (number|null)[] = [0, 1, 2]) {
  return [
    values.map((x: number|null) => x == null ? null : Number(x) + 0),
    values.map((x: number|null) => x == null ? null : Number(x) + 1),
  ] as [(number | null)[], (number | null)[]];
}

export function makeTestBigInts(values: (number|bigint|null)[] = [0, 1, 2]) {
  return [
    values.map((x: number|bigint|null) => x == null ? null : BigInt(x) + 0n),
    values.map((x: number|bigint|null) => x == null ? null : BigInt(x) + 1n),
  ] as [(bigint | null)[], (bigint | null)[]];
}

export function makeTestSeries<T extends arrow.DataType>(
  type: T, [lhs, rhs]: [(number | bigint | null)[], (number | bigint | null)[]]) {
  return {
    lhs: Series.new(arrow.Vector.from({type, values: lhs})),
    rhs: Series.new(arrow.Vector.from({type, values: rhs})),
  };
}

export type MathematicalUnaryOp = 'sin'|'cos'|'tan'|'asin'|'acos'|'atan'|'sinh'|'cosh'|'tanh'|
  'asinh'|'acosh'|'atanh'|'exp'|'log'|'sqrt'|'cbrt'|'ceil'|'floor'|'abs';

export const mathematicalUnaryOps: MathematicalUnaryOp[] = [
  'sin',
  'cos',
  'tan',
  'asin',
  'acos',
  'atan',
  'sinh',
  'cosh',
  'tanh',
  'asinh',
  'acosh',
  'atanh',
  'exp',
  'log',
  'sqrt',
  'cbrt',
  'ceil',
  'floor',
  'abs'
];

export const clampIntValuesLikeUnaryCast =
  (a: Int8Array|Int16Array|Int32Array|Uint8Array|Uint16Array|Uint32Array) =>
    <T extends DataType>(type: T, input: number[]) => {
      return input.map((x) => {
        a[0] = x;
        if (type instanceof Bool8) { return a[0] ? 1 : 0; }
        if (type instanceof Int64) { return BigInt.asIntN(64, BigInt(a[0])); }
        if (type instanceof Uint64) { return BigInt.asUintN(64, BigInt(a[0])); }
        return a[0];
      });
    };

export function clampFloatValuesLikeUnaryCast<T extends DataType>(type: T, input: number[]) {
  return input.map((x) => {
    if (type instanceof Bool8) { return x ? 1 : 0; }
    if ((type instanceof Int8) || (type instanceof Int16) || (type instanceof Int32)) {
      return x | 0;
    }
    if (type instanceof Int64) { return BigInt.asIntN(64, BigInt(x | 0)); }
    if ((type instanceof Uint8) || (type instanceof Uint16) || (type instanceof Uint32)) {
      return x < 0 ? 0 : x | 0;
    }
    if (type instanceof Uint64) { return BigInt.asUintN(64, BigInt(x < 0 ? 0 : x | 0)); }
    return x;
  });
}

export const testForEachNumericType =
  (name: string,
   fn: (() => void)|(<T extends TypedArray|BigIntArray, R extends Numeric>(
                       TypedArrayCtor: TypedArrayConstructor<T>, type: R) => void)) =>
    test.each([
      [Int8Array, new Int8],
      [Int16Array, new Int16],
      [Int32Array, new Int32],
      [BigInt64Array, new Int64],
      [Uint8Array, new Uint8],
      [Uint16Array, new Uint16],
      [Uint32Array, new Uint32],
      [BigUint64Array, new Uint64],
      [Float32Array, new Float32],
      [Float64Array, new Float64],
      [Uint8ClampedArray, new Bool8],
    ])(name, (TypedArrayCtor: any, type: any) => fn(TypedArrayCtor, type));
