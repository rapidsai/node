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
  Float32Buffer,
  Float64Buffer,
  Int16Buffer,
  Int32Buffer,
  Int64Buffer,
  Int8Buffer,
  MemoryViewConstructor,
  TypedArray,
  TypedArrayConstructor,
  Uint16Buffer,
  Uint32Buffer,
  Uint64Buffer,
  Uint8Buffer,
  Uint8ClampedBuffer
} from '@nvidia/cuda';
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
  TypeId,
  Uint16,
  Uint32,
  Uint64,
  Uint8
} from '@nvidia/cudf';
import * as arrow from 'apache-arrow';

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
        switch (type.id) {
          case TypeId.BOOL8: return a[0] ? 1 : 0;
          case TypeId.INT64: return BigInt.asIntN(64, BigInt(a[0]));
          case TypeId.UINT64: return BigInt.asUintN(64, BigInt(a[0]));
          default: return a[0];
        }
      });
    };

export function clampFloatValuesLikeUnaryCast<T extends DataType>(type: T, input: number[]) {
  return input.map((x) => {
    switch (type.id) {
      case TypeId.BOOL8: return x ? 1 : 0;
      case TypeId.INT8:
      case TypeId.INT16:
      case TypeId.INT32: return x | 0;
      case TypeId.INT64: return BigInt.asIntN(64, BigInt(x | 0));
      case TypeId.UINT8:
      case TypeId.UINT16:
      case TypeId.UINT32: return x < 0 ? 0 : x | 0;
      case TypeId.UINT64: return BigInt.asUintN(64, BigInt(x < 0 ? 0 : x | 0));
      default: return x;
    }
  });
}

export const testForEachNumericType =
  (name: string,
   fn: (() => void)|
   (<T extends TypedArray|BigIntArray, R extends Numeric>(TypedArrayCtor: TypedArrayConstructor<T>,
                                                          MemoryViewCtor: MemoryViewConstructor<T>,
                                                          type: R) => void)) =>
    test.each([
      [Int8Array, Int8Buffer, new Int8],
      [Int16Array, Int16Buffer, new Int16],
      [Int32Array, Int32Buffer, new Int32],
      [BigInt64Array, Int64Buffer, new Int64],
      [Uint8Array, Uint8Buffer, new Uint8],
      [Uint16Array, Uint16Buffer, new Uint16],
      [Uint32Array, Uint32Buffer, new Uint32],
      [BigUint64Array, Uint64Buffer, new Uint64],
      [Float32Array, Float32Buffer, new Float32],
      [Float64Array, Float64Buffer, new Float64],
      [Uint8ClampedArray, Uint8ClampedBuffer, new Bool8],
    ])(name,
       (TypedArrayCtor: any, MemoryViewCtor: any, type: any) =>
         fn(TypedArrayCtor, MemoryViewCtor, type));
