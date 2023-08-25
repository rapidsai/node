// Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
} from '@rapidsai/cuda';
import {MemoryResource} from '@rapidsai/rmm';

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
  Utf8String
} from '../types/dtypes';
import {CommonType} from '../types/mappings';

import {Float64Series} from './float';
import {NumericSeries} from './numeric';
import {StringSeries} from './string';

/**
 * A base class for Series of 8, 16, 32, or 64-bit integral values in GPU memory.
 */
abstract class IntSeries<T extends Integral> extends NumericSeries<T> {
  _castAsString(memoryResource?: MemoryResource): StringSeries {
    return StringSeries.new(this._col.stringsFromIntegers(memoryResource));
  }

  /**
   * Returns a new string Series converting integer columns to hexadecimal characters.
   *
   * Any null entries will result in corresponding null entries in the output series.
   *
   * The output character set is '0'-'9' and 'A'-'F'. The output string width will be a multiple of
   * 2 depending on the size of the integer type. A single leading zero is applied to the first
   * non-zero output byte if it less than 0x10.
   *
   * Leading zeros are suppressed unless filling out a complete byte as in 1234 -> 04D2 instead of
   * 000004D2 or 4D2.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Series' device
   *   memory.
   */
  toHexString(memoryResource?: MemoryResource): Series<Utf8String> {
    return Series.new(this._col.hexFromIntegers(memoryResource));
  }

  /**
   * Perform a binary `&` operation between this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to use.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  bitwiseAnd(rhs: bigint, memoryResource?: MemoryResource): Int64Series;
  bitwiseAnd(rhs: number, memoryResource?: MemoryResource): Float64Series;
  bitwiseAnd<R extends Scalar<Integral>>(rhs: R, memoryResource?: MemoryResource):
    Series<CommonType<T, R['type']>>;
  bitwiseAnd<R extends Series<Integral>>(rhs: R, memoryResource?: MemoryResource):
    Series<CommonType<T, R['type']>>;
  bitwiseAnd<R extends Integral>(rhs: bigint|number|Scalar<R>|Series<R>,
                                 memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.bitwiseAnd(rhs, memoryResource));
      case 'number': return Series.new(this._col.bitwiseAnd(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar
           ? Series.new(this._col.bitwiseAnd(rhs, memoryResource))
           : Series.new(this._col.bitwiseAnd(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Perform a binary `|` operation between this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to use.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  bitwiseOr(rhs: bigint, memoryResource?: MemoryResource): Int64Series;
  bitwiseOr(rhs: number, memoryResource?: MemoryResource): Float64Series;
  bitwiseOr<R extends Scalar<Integral>>(rhs: R, memoryResource?: MemoryResource):
    Series<CommonType<T, R['type']>>;
  bitwiseOr<R extends Series<Integral>>(rhs: R, memoryResource?: MemoryResource):
    Series<CommonType<T, R['type']>>;
  bitwiseOr<R extends Integral>(rhs: bigint|number|Scalar<R>|Series<R>,
                                memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.bitwiseOr(rhs, memoryResource));
      case 'number': return Series.new(this._col.bitwiseOr(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar
           ? Series.new(this._col.bitwiseOr(rhs, memoryResource))
           : Series.new(this._col.bitwiseOr(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Perform a binary `^` operation between this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to use.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  bitwiseXor(rhs: bigint, memoryResource?: MemoryResource): Int64Series;
  bitwiseXor(rhs: number, memoryResource?: MemoryResource): Float64Series;
  bitwiseXor<R extends Scalar<Integral>>(rhs: R, memoryResource?: MemoryResource):
    Series<CommonType<T, R['type']>>;
  bitwiseXor<R extends Series<Integral>>(rhs: R, memoryResource?: MemoryResource):
    Series<CommonType<T, R['type']>>;
  bitwiseXor<R extends Integral>(rhs: bigint|number|Scalar<R>|Series<R>,
                                 memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.bitwiseXor(rhs, memoryResource));
      case 'number': return Series.new(this._col.bitwiseXor(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar
           ? Series.new(this._col.bitwiseXor(rhs, memoryResource))
           : Series.new(this._col.bitwiseXor(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Perform a binary `<<` operation between this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to use.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  shiftLeft(rhs: bigint, memoryResource?: MemoryResource): Int64Series;
  shiftLeft(rhs: number, memoryResource?: MemoryResource): Float64Series;
  shiftLeft<R extends Scalar<Integral>>(rhs: R, memoryResource?: MemoryResource):
    Series<CommonType<T, R['type']>>;
  shiftLeft<R extends Series<Integral>>(rhs: R, memoryResource?: MemoryResource):
    Series<CommonType<T, R['type']>>;
  shiftLeft<R extends Integral>(rhs: bigint|number|Scalar<R>|Series<R>,
                                memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.shiftLeft(rhs, memoryResource));
      case 'number': return Series.new(this._col.shiftLeft(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar
           ? Series.new(this._col.shiftLeft(rhs, memoryResource))
           : Series.new(this._col.shiftLeft(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Perform a binary `>>` operation between this Series and another Series or scalar
   * value.
   *
   * @param rhs The other Series or scalar to use.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  shiftRight(rhs: bigint, memoryResource?: MemoryResource): Int64Series;
  shiftRight(rhs: number, memoryResource?: MemoryResource): Float64Series;
  shiftRight<R extends Scalar<Integral>>(rhs: R, memoryResource?: MemoryResource):
    Series<CommonType<T, R['type']>>;
  shiftRight<R extends Series<Integral>>(rhs: R, memoryResource?: MemoryResource):
    Series<CommonType<T, R['type']>>;
  shiftRight<R extends Integral>(rhs: bigint|number|Scalar<R>|Series<R>,
                                 memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.shiftRight(rhs, memoryResource));
      case 'number': return Series.new(this._col.shiftRight(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar
           ? Series.new(this._col.shiftRight(rhs, memoryResource))
           : Series.new(this._col.shiftRight(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Perform a binary `shiftRightUnsigned` operation between this Series and another Series or
   * scalar value.
   *
   * @param rhs The other Series or scalar to use.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  shiftRightUnsigned(rhs: bigint, memoryResource?: MemoryResource): Int64Series;
  shiftRightUnsigned(rhs: number, memoryResource?: MemoryResource): Float64Series;
  shiftRightUnsigned<R extends Scalar<Integral>>(rhs: R, memoryResource?: MemoryResource):
    Series<CommonType<T, R['type']>>;
  shiftRightUnsigned<R extends Series<Integral>>(rhs: R, memoryResource?: MemoryResource):
    Series<CommonType<T, R['type']>>;
  shiftRightUnsigned<R extends Integral>(rhs: bigint|number|Scalar<R>|Series<R>,
                                         memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.shiftRightUnsigned(rhs, memoryResource));
      case 'number': return Series.new(this._col.shiftRightUnsigned(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar
           ? Series.new(this._col.shiftRightUnsigned(rhs, memoryResource))
           : Series.new(this._col.shiftRightUnsigned(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Compute the bitwise not (~) for each value in this Series.
   *
   * @param memoryResource Memory resource used to allocate the result Series's device memory.
   * @returns A Series of the same number of elements containing the result of the operation.
   */
  bitInvert(memoryResource?: MemoryResource): Series<T> {
    return this.__construct(this._col.bitInvert(memoryResource));
  }

  /**
   * Compute the cumulative max of all values in this Series.
   *
   * @param skipNulls The optional skipNulls if true drops NA and null values before computing
   *   reduction,
   * else if skipNulls is false, reduction is computed directly.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns The cumulative max of all the values in this Series.
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([4, 2, 5, 1, 1]).cast(new Int32)
   *
   * a.cumulativeMax() // {4, 4, 5, 5, 5}
   * ```
   */
  cumulativeMax(skipNulls = true, memoryResource?: MemoryResource): Series<T> {
    return this.__construct(this._prepare_scan_series(skipNulls).cumulativeMax(memoryResource));
  }

  /**
   * Compute the cumulative min of all values in this Series.
   *
   * @param skipNulls The optional skipNulls if true drops NA and null values before computing
   *   reduction,
   * else if skipNulls is false, reduction is computed directly.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns The cumulative min of all the values in this Series.
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([4, 2, 5, 1, 1]).cast(new Int32)
   *
   * a.cumulativeMin() // {4, 2, 2, 1, 1}
   * ```
   */
  cumulativeMin(skipNulls = true, memoryResource?: MemoryResource): Series<T> {
    return this.__construct(this._prepare_scan_series(skipNulls).cumulativeMin(memoryResource));
  }

  /**
   * Compute the cumulative product of all values in this Series.
   *
   * @param skipNulls The optional skipNulls if true drops NA and null values before computing
   *   reduction,
   * else if skipNulls is false, reduction is computed directly.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns The cumulative product of all the values in this Series.
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([4, 2, 5, 1, 1]).cast(new Int32)
   *
   * a.cumulativeProduct() // {4, 8, 40, 40, 40}
   * ```
   */
  cumulativeProduct(skipNulls = true, memoryResource?: MemoryResource): Series<T> {
    return this.__construct(this._prepare_scan_series(skipNulls).cumulativeProduct(memoryResource));
  }

  /**
   * Compute the cumulative sum of all values in this Series.
   *
   * @param skipNulls The optional skipNulls if true drops NA and null values before computing
   *   reduction,
   * else if skipNulls is false, reduction is computed directly.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns The cumulative sum of all the values in this Series.
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([4, 2, 5, 1, 1]).cast(new Int32)
   *
   * a.cumulativeSum() // {4, 6, 11, 12, 13}
   * ```
   */
  cumulativeSum(skipNulls = true, memoryResource?: MemoryResource): Series<T> {
    return this.__construct(this._prepare_scan_series(skipNulls).cumulativeSum(memoryResource));
  }

  /** @inheritdoc */
  sum(skipNulls = true, memoryResource?: MemoryResource) {
    return super.sum(skipNulls, memoryResource) as bigint;
  }

  /** @inheritdoc */
  product(skipNulls = true, memoryResource?: MemoryResource) {
    return super.product(skipNulls, memoryResource) as bigint;
  }

  /** @inheritdoc */
  sumOfSquares(skipNulls = true, memoryResource?: MemoryResource) {
    return super.sumOfSquares(skipNulls, memoryResource) as bigint;
  }
}

/**
 * A Series of 8-bit signed integer values in GPU memory.
 */
export class Int8Series extends IntSeries<Int8> {
  /**
   * A Int8 view of the values in GPU memory.
   */
  get data() {
    return new Int8Buffer(this._col.data).subarray(this.offset, this.offset + this.length);
  }

  /** @inheritdoc */
  min(skipNulls = true, memoryResource?: MemoryResource) {
    return super.min(skipNulls, memoryResource) as number;
  }

  /** @inheritdoc */
  max(skipNulls = true, memoryResource?: MemoryResource) {
    return super.max(skipNulls, memoryResource) as number;
  }

  /** @inheritdoc */
  minmax(skipNulls = true, memoryResource?: MemoryResource) {
    return super.minmax(skipNulls, memoryResource) as [number, number];
  }
}

/**
 * A Series of 16-bit signed integer values in GPU memory.
 */
export class Int16Series extends IntSeries<Int16> {
  /**
   * A Int16 view of the values in GPU memory.
   */
  get data() {
    return new Int16Buffer(this._col.data).subarray(this.offset, this.offset + this.length);
  }

  /** @inheritdoc */
  min(skipNulls = true, memoryResource?: MemoryResource) {
    return super.min(skipNulls, memoryResource) as number;
  }

  /** @inheritdoc */
  max(skipNulls = true, memoryResource?: MemoryResource) {
    return super.max(skipNulls, memoryResource) as number;
  }

  /** @inheritdoc */
  minmax(skipNulls = true, memoryResource?: MemoryResource) {
    return super.minmax(skipNulls, memoryResource) as [number, number];
  }
}

/**
 * A Series of 32-bit signed integer values in GPU memory.
 */
export class Int32Series extends IntSeries<Int32> {
  /**
   * A Int32 view of the values in GPU memory.
   */
  get data() {
    return new Int32Buffer(this._col.data).subarray(this.offset, this.offset + this.length);
  }

  /** @inheritdoc */
  min(skipNulls = true, memoryResource?: MemoryResource) {
    return super.min(skipNulls, memoryResource) as number;
  }

  /** @inheritdoc */
  max(skipNulls = true, memoryResource?: MemoryResource) {
    return super.max(skipNulls, memoryResource) as number;
  }

  /** @inheritdoc */
  minmax(skipNulls = true, memoryResource?: MemoryResource) {
    return super.minmax(skipNulls, memoryResource) as [number, number];
  }
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
  get data() {
    return new Int64Buffer(this._col.data).subarray(this.offset, this.offset + this.length);
  }

  /**
   * Converts integers into IPv4 addresses as strings.
   *
   * The IPv4 format is 1-3 character digits [0-9] between 3 dots (e.g. 123.45.67.890). Each section
   * can have a value between [0-255].
   *
   * Each input integer is dissected into four integers by dividing the input into 8-bit sections.
   * These sub-integers are then converted into [0-9] characters and placed between '.' characters.
   *
   * No checking is done on the input integer value. Only the lower 32-bits are used.
   *
   * Any null entries will result in corresponding null entries in the output series.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Series' device
   *   memory.
   */
  toIpv4String(memoryResource?: MemoryResource): Series<Utf8String> {
    return Series.new(this._col.ipv4FromIntegers(memoryResource));
  }

  /** @inheritdoc */
  min(skipNulls = true, memoryResource?: MemoryResource) {
    return super.min(skipNulls, memoryResource) as bigint;
  }

  /** @inheritdoc */
  max(skipNulls = true, memoryResource?: MemoryResource) {
    return super.max(skipNulls, memoryResource) as bigint;
  }

  /** @inheritdoc */
  minmax(skipNulls = true, memoryResource?: MemoryResource) {
    return super.minmax(skipNulls, memoryResource) as [bigint, bigint];
  }
}

/**
 * A Series of 8-bit unsigned integer values in GPU memory.
 */
export class Uint8Series extends IntSeries<Uint8> {
  /**
   * A Uint8 view of the values in GPU memory.
   */
  get data() {
    return new Uint8Buffer(this._col.data).subarray(this.offset, this.offset + this.length);
  }

  /** @inheritdoc */
  min(skipNulls = true, memoryResource?: MemoryResource) {
    return super.min(skipNulls, memoryResource) as number;
  }

  /** @inheritdoc */
  max(skipNulls = true, memoryResource?: MemoryResource) {
    return super.max(skipNulls, memoryResource) as number;
  }

  /** @inheritdoc */
  minmax(skipNulls = true, memoryResource?: MemoryResource) {
    return super.minmax(skipNulls, memoryResource) as [number, number];
  }
}

/**
 * A Series of 16-bit unsigned integer values in GPU memory.
 */
export class Uint16Series extends IntSeries<Uint16> {
  /**
   * A Uint16 view of the values in GPU memory.
   */
  get data() {
    return new Uint16Buffer(this._col.data).subarray(this.offset, this.offset + this.length);
  }

  /** @inheritdoc */
  min(skipNulls = true, memoryResource?: MemoryResource) {
    return super.min(skipNulls, memoryResource) as number;
  }

  /** @inheritdoc */
  max(skipNulls = true, memoryResource?: MemoryResource) {
    return super.max(skipNulls, memoryResource) as number;
  }

  /** @inheritdoc */
  minmax(skipNulls = true, memoryResource?: MemoryResource) {
    return super.minmax(skipNulls, memoryResource) as [number, number];
  }
}

/**
 * A Series of 32-bit unsigned integer values in GPU memory.
 */
export class Uint32Series extends IntSeries<Uint32> {
  /**
   * A Uint32 view of the values in GPU memory.
   */
  get data() {
    return new Uint32Buffer(this._col.data).subarray(this.offset, this.offset + this.length);
  }

  /** @inheritdoc */
  min(skipNulls = true, memoryResource?: MemoryResource) {
    return super.min(skipNulls, memoryResource) as number;
  }

  /** @inheritdoc */
  max(skipNulls = true, memoryResource?: MemoryResource) {
    return super.max(skipNulls, memoryResource) as number;
  }

  /** @inheritdoc */
  minmax(skipNulls = true, memoryResource?: MemoryResource) {
    return super.minmax(skipNulls, memoryResource) as [number, number];
  }
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
  get data() {
    return new Uint64Buffer(this._col.data).subarray(this.offset, this.offset + this.length);
  }

  /** @inheritdoc */
  min(skipNulls = true, memoryResource?: MemoryResource) {
    return super.min(skipNulls, memoryResource) as bigint;
  }

  /** @inheritdoc */
  max(skipNulls = true, memoryResource?: MemoryResource) {
    return super.max(skipNulls, memoryResource) as bigint;
  }

  /** @inheritdoc */
  minmax(skipNulls = true, memoryResource?: MemoryResource) {
    return super.minmax(skipNulls, memoryResource) as [bigint, bigint];
  }
}
