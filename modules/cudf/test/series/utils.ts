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

import {Series} from '@nvidia/cudf';
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
