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
  Int16Buffer,
  Int32Buffer,
  Int64Buffer,
  Int8Buffer,
  Uint16Buffer,
  Uint32Buffer,
  Uint64Buffer,
  Uint8Buffer,
} from '@nvidia/cuda';
import {MemoryResource} from '@nvidia/rmm';

import {Column} from '../column';
import {Scalar} from '../scalar';
import {Series} from '../series';
import {
  Int16,
  Int32,
  Int64,
  Int8,
  Integral,
  Uint16,
  Uint32,
  Uint64,
  Uint8,
} from '../types/dtypes'

import {NumericSeries} from './numeric';

type Integer = Integral|Int64|Uint64;

/**
 * A base class for Series of 8, 16, 32, or 64-bit integral values in GPU memory.
 */
abstract class IntSeries<T extends Integer> extends NumericSeries<T> {
  /**
   * Perform a binary `&` operation between this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to use.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  bitwise_and(rhs: bigint, memoryResource?: MemoryResource): Series<T>;
  bitwise_and(rhs: number, memoryResource?: MemoryResource): Series<T>;
  bitwise_and<R extends Integer>(rhs: Scalar<R>, memoryResource?: MemoryResource): Series<T>;
  bitwise_and<R extends Integer>(rhs: IntSeries<R>, memoryResource?: MemoryResource): Series<T>;
  bitwise_and<R extends Integer>(rhs: bigint|number|Scalar<R>|Series<R>,
                                 memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.bitwise_and(rhs, memoryResource));
      case 'number': return Series.new(this._col.bitwise_and(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar
             ? Series.new(this._col.bitwise_and(rhs, memoryResource))
             : Series.new(this._col.bitwise_and(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Perform a binary `|` operation between this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to use.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  bitwise_or(rhs: bigint, memoryResource?: MemoryResource): Series<T>;
  bitwise_or(rhs: number, memoryResource?: MemoryResource): Series<T>;
  bitwise_or<R extends Integer>(rhs: Scalar<R>, memoryResource?: MemoryResource): Series<T>;
  bitwise_or<R extends Integer>(rhs: IntSeries<R>, memoryResource?: MemoryResource): Series<T>;
  bitwise_or<R extends Integer>(rhs: bigint|number|Scalar<R>|Series<R>,
                                memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.bitwise_or(rhs, memoryResource));
      case 'number': return Series.new(this._col.bitwise_or(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar
             ? Series.new(this._col.bitwise_or(rhs, memoryResource))
             : Series.new(this._col.bitwise_or(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Perform a binary `^` operation between this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to use.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  bitwise_xor(rhs: bigint, memoryResource?: MemoryResource): Series<T>;
  bitwise_xor(rhs: number, memoryResource?: MemoryResource): Series<T>;
  bitwise_xor<R extends Integer>(rhs: Scalar<R>, memoryResource?: MemoryResource): Series<T>;
  bitwise_xor<R extends Integer>(rhs: IntSeries<R>, memoryResource?: MemoryResource): Series<T>;
  bitwise_xor<R extends Integer>(rhs: bigint|number|Scalar<R>|Series<R>,
                                 memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.bitwise_xor(rhs, memoryResource));
      case 'number': return Series.new(this._col.bitwise_xor(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar
             ? Series.new(this._col.bitwise_xor(rhs, memoryResource))
             : Series.new(this._col.bitwise_xor(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Perform a binary `<<` operation between this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to use.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  shift_left(rhs: bigint, memoryResource?: MemoryResource): Series<T>;
  shift_left(rhs: number, memoryResource?: MemoryResource): Series<T>;
  shift_left<R extends Integer>(rhs: Scalar<R>, memoryResource?: MemoryResource): Series<T>;
  shift_left<R extends Integer>(rhs: IntSeries<R>, memoryResource?: MemoryResource): Series<T>;
  shift_left<R extends Integer>(rhs: bigint|number|Scalar<R>|Series<R>,
                                memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.shift_left(rhs, memoryResource));
      case 'number': return Series.new(this._col.shift_left(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar
             ? Series.new(this._col.shift_left(rhs, memoryResource))
             : Series.new(this._col.shift_left(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Perform a binary `>>` operation between this Series and another Series or scalar
   * value.
   *
   * @param rhs The other Series or scalar to use.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  shift_right(rhs: bigint, memoryResource?: MemoryResource): Series<T>;
  shift_right(rhs: number, memoryResource?: MemoryResource): Series<T>;
  shift_right<R extends Integer>(rhs: Scalar<R>, memoryResource?: MemoryResource): Series<T>;
  shift_right<R extends Integer>(rhs: IntSeries<R>, memoryResource?: MemoryResource): Series<T>;
  shift_right<R extends Integer>(rhs: bigint|number|Scalar<R>|Series<R>,
                                 memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.shift_right(rhs, memoryResource));
      case 'number': return Series.new(this._col.shift_right(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar
             ? Series.new(this._col.shift_right(rhs, memoryResource))
             : Series.new(this._col.shift_right(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Perform a binary `shift_right_unsigned` operation between this Series and another Series or
   * scalar value.
   *
   * @param rhs The other Series or scalar to use.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  shift_right_unsigned(rhs: bigint, memoryResource?: MemoryResource): Series<T>;
  shift_right_unsigned(rhs: number, memoryResource?: MemoryResource): Series<T>;
  shift_right_unsigned<R extends Integer>(rhs: Scalar<R>,
                                          memoryResource?: MemoryResource): Series<T>;
  shift_right_unsigned<R extends Integer>(rhs: IntSeries<R>,
                                          memoryResource?: MemoryResource): Series<T>;
  shift_right_unsigned<R extends Integer>(rhs: bigint|number|Scalar<R>|Series<R>,
                                          memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.shift_right_unsigned(rhs, memoryResource));
      case 'number': return Series.new(this._col.shift_right_unsigned(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar
             ? Series.new(this._col.shift_right_unsigned(rhs, memoryResource))
             : Series.new(this._col.shift_right_unsigned(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Compute the bitwise not (~) for each value in this Series.
   *
   * @param memoryResource Memory resource used to allocate the result Series's device memory.
   * @returns A Series of the same number of elements containing the result of the operation.
   */
  bit_invert(memoryResource?: MemoryResource): Series<T> {
    return Series.new(this._col.bit_invert(memoryResource));
  }
}

/**
 * A Series of 8-bit signed integer values in GPU memory.
 */
export class Int8Series extends IntSeries<Int8> {
  /**
   * A Int8 view of the values in GPU memory.
   */
  get data() { return new Int8Buffer(this._col.data).subarray(0, this.length); }
}

/**
 * A Series of 16-bit signed integer values in GPU memory.
 */
export class Int16Series extends IntSeries<Int16> {
  /**
   * A Int16 view of the values in GPU memory.
   */
  get data() { return new Int16Buffer(this._col.data).subarray(0, this.length); }
}

/**
 * A Series of 32-bit signed integer values in GPU memory.
 */
export class Int32Series extends IntSeries<Int32> {
  /**
   * A Int32 view of the values in GPU memory.
   */
  get data() { return new Int32Buffer(this._col.data).subarray(0, this.length); }
}

/**
 * A Series of 64-bit signed integer values in GPU memory.
 */
export class Int64Series extends IntSeries<Int64> {
  * [Symbol.iterator]() {
    for (const x of super[Symbol.iterator]()) { yield x === null ? x : 0n + x; }
  }
  /**
   * A Int64 view of the values in GPU memory.
   */
  get data() { return new Int64Buffer(this._col.data).subarray(0, this.length); }
}

/**
 * A Series of 8-bit unsigned integer values in GPU memory.
 */
export class Uint8Series extends IntSeries<Uint8> {
  /**
   * A Uint8 view of the values in GPU memory.
   */
  get data() { return new Uint8Buffer(this._col.data).subarray(0, this.length); }
}

/**
 * A Series of 16-bit unsigned integer values in GPU memory.
 */
export class Uint16Series extends IntSeries<Uint16> {
  /**
   * A Uint16 view of the values in GPU memory.
   */
  get data() { return new Uint16Buffer(this._col.data).subarray(0, this.length); }
}

/**
 * A Series of 32-bit unsigned integer values in GPU memory.
 */
export class Uint32Series extends IntSeries<Uint32> {
  /**
   * A Uint32 view of the values in GPU memory.
   */
  get data() { return new Uint32Buffer(this._col.data).subarray(0, this.length); }
}

/**
 * A Series of 64-bit unsigned integer values in GPU memory.
 */
export class Uint64Series extends IntSeries<Uint64> {
  * [Symbol.iterator]() {
    for (const x of super[Symbol.iterator]()) { yield x === null ? x : 0n + x; }
  }
  /**
   * A Uint64 view of the values in GPU memory.
   */
  get data() { return new Uint64Buffer(this._col.data).subarray(0, this.length); }
}
