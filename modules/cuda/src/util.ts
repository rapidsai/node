// Copyright (c) 2020, NVIDIA CORPORATION.
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

import {Memory} from './memory';

/** @ignore */
export const isNumber = (x: any): x is number => typeof x === 'number';
/** @ignore */
export const isBigInt = (x: any): x is bigint => typeof x === 'bigint';
/** @ignore */
export const isBoolean = (x: any): x is boolean => typeof x === 'boolean';
/** @ignore */
// eslint-disable-next-line @typescript-eslint/ban-types
export const isFunction = (x: any): x is Function => typeof x === 'function';
/** @ignore */
// eslint-disable-next-line @typescript-eslint/ban-types
export const isObject = (x: any): x is Object => x != null && Object(x) === x;

/** @ignore */
export const isPromise =
  <T = any>(x: any): x is PromiseLike<T> => { return isObject(x) && isFunction(x.then);};

/** @ignore */
export const isIterable =
  <T = any>(x: any): x is Iterable<T> => { return isObject(x) && isFunction(x[Symbol.iterator]);};

/** @ignore */
export const isAsyncIterable = <T = any>(x: any):
  x is AsyncIterable<T>      => { return isObject(x) && isFunction(x[Symbol.asyncIterator]);};

/** @ignore */
export const isArrayLike =
  <T = any>(x: any): x is ArrayLike<T> => { return isObject(x) && isNumber(x.length);};

/** @ignore */
export const isMemoryLike =
  (x: any): x is Memory => { return isObject(x) && isNumber(x.ptr) && isNumber(x.byteLength);};

/** @ignore */
export const isArrayBufferLike = (x: any): x is ArrayBufferLike => {
  switch (x && x.constructor && x.constructor.name) {
    case 'ArrayBuffer': return true;
    case 'SharedArrayBuffer': return true;
    default: return false;
  }
};

/** @ignore */
// eslint-disable-next-line @typescript-eslint/unbound-method
export const isArrayBufferView = ArrayBuffer.isView;

/** @ignore */
export const isIteratorResult = <T = any>(x: any):
  x is IteratorResult<T>      => { return isObject(x) && ('done' in x) && ('value' in x);};

/**
 * @summary Clamp begin and end ranges similar to `Array.prototype.slice`.
 * @description Normalizes begin/end to between 0 and length, and wrap around on negative indices.
 * @example
 * ```typescript
 * import {clampRange} from '@nvidia/cuda';
 *
 * clampRange(5)        // [0, 5]
 * clampRange(5, 0, -1) // [0, 4]
 * clampRange(5, -1)    // [4, 5]
 * clampRange(5, -1, 0) // [4, 4]
 *
 * const ary = Array.from({length: 5}, (_, i) => i);
 * // [0, 1, 2, 3, 4]
 * assert(ary.slice() == ary.slice(...clampRange(ary.length)))
 * // [0, 1, 2, 3]
 * assert(ary.slice(0, -1) == ary.slice(...clampRange(ary.length, 0, -1)))
 * // [4]
 * assert(ary.slice(-1) == ary.slice(...clampRange(ary.length, -1)))
 * // []
 * assert(ary.slice(-1, 0) == ary.slice(...clampRange(ary.length, -1, 0)))
 * ```
 *
 * @param len The total number of elements.
 * @param lhs The beginning of the range to clamp.
 * @param rhs The end of the range to clamp (<b>Default:</b> `len`).
 * @returns An Array of the normalized begin and end positions.
 */
export function clampRange(len: number, lhs = 0, rhs = len): [begin: number, end: number] {
  // wrap around on negative begin and end positions
  if (lhs < 0) { lhs = ((lhs % len) + len) % len; }
  if (rhs < 0) { rhs = ((rhs % len) + len) % len; }
  // enforce lhs <= rhs && lhs <= len && rhs <= len
  return rhs < lhs ? [lhs > len ? len : lhs, lhs > len ? len : lhs] : [lhs, rhs > len ? len : rhs];
}
