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

import {MemoryResource} from '@nvidia/rmm';

import {Column} from '../column';
import {Scalar} from '../scalar';
import {Series} from '../series';
import {Bool8, DataType, Numeric} from '../types/dtypes'
import {CommonType, Interpolation} from '../types/mappings';

import {Float64Series} from './float';
import {Int64Series} from './integral';

/**
 * A base class for Series of fixed-width numeric values.
 */
export abstract class NumericSeries<T extends Numeric> extends Series<T> {
  /**
   * Casts the values to a new dtype (similar to `static_cast` in C++).
   *
   * @param dataType The new dtype.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns Series of same size as the current Series containing result of the `cast` operation.
   */
  cast<R extends DataType>(dataType: R, memoryResource?: MemoryResource): Series<R> {
    return Series.new(this._col.cast(dataType, memoryResource));
  }

  /**
   * View the data underlying this Series as a new dtype (similar to `reinterpret_cast` in C++).
   *
   * @note The length of this Series must divide evenly into the size of the desired data type.
   * @note Series with nulls may only be viewed as dtypes of the same element width.
   *
   * @returns Series of same size as the current Series containing result of the `cast` operation.
   */
  view<R extends Numeric>(dataType: R): Series<R> {
    if (this.type.BYTES_PER_ELEMENT === dataType.BYTES_PER_ELEMENT) {
      return Series.new({
        type: dataType,
        data: this._col.data,
        length: this.length,
        nullMask: this.mask,
        nullCount: this.nullCount,
      });
    }
    if (this.nullCount > 0) {
      throw new Error('Cannot view a Series with nulls as a dtype of a different element width');
    }
    const byteLength = this.length * this.type.BYTES_PER_ELEMENT;
    if (0 !== (byteLength % dataType.BYTES_PER_ELEMENT)) {
      throw new Error(
        `Can not divide ${this.length * this.type.BYTES_PER_ELEMENT} total bytes into ${
          String(dataType)} with element width ${dataType.BYTES_PER_ELEMENT}`);
    }
    const newLength = byteLength / dataType.BYTES_PER_ELEMENT;
    return Series.new({type: dataType, data: this._col.data, length: newLength});
  }

  /**
   * Add this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to add to this Series.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  add(rhs: bigint, memoryResource?: MemoryResource): Int64Series;
  add(rhs: number, memoryResource?: MemoryResource): Float64Series;
  add<R extends Numeric>(rhs: Scalar<R>, memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  add<R extends Numeric>(rhs: NumericSeries<R>,
                         memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  add<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>, memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.add(rhs, memoryResource));
      case 'number': return Series.new(this._col.add(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar ? Series.new(this._col.add(rhs, memoryResource))
                                 : Series.new(this._col.add(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Subtract this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to subtract from this Series.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  sub(rhs: bigint, memoryResource?: MemoryResource): Int64Series;
  sub(rhs: number, memoryResource?: MemoryResource): Float64Series;
  sub<R extends Numeric>(rhs: Scalar<R>, memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  sub<R extends Numeric>(rhs: NumericSeries<R>,
                         memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  sub<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>, memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.sub(rhs, memoryResource));
      case 'number': return Series.new(this._col.sub(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar ? Series.new(this._col.sub(rhs, memoryResource))
                                 : Series.new(this._col.sub(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Multiply this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to multiply this column by.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  mul(rhs: bigint, memoryResource?: MemoryResource): Int64Series;
  mul(rhs: number, memoryResource?: MemoryResource): Float64Series;
  mul<R extends Numeric>(rhs: Scalar<R>, memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  mul<R extends Numeric>(rhs: NumericSeries<R>,
                         memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  mul<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>, memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.mul(rhs, memoryResource));
      case 'number': return Series.new(this._col.mul(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar ? Series.new(this._col.mul(rhs, memoryResource))
                                 : Series.new(this._col.mul(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Divide this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to divide this Series by.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  div(rhs: bigint, memoryResource?: MemoryResource): Int64Series;
  div(rhs: number, memoryResource?: MemoryResource): Float64Series;
  div<R extends Numeric>(rhs: Scalar<R>, memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  div<R extends Numeric>(rhs: NumericSeries<R>,
                         memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  div<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>, memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.div(rhs, memoryResource));
      case 'number': return Series.new(this._col.div(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar ? Series.new(this._col.div(rhs, memoryResource))
                                 : Series.new(this._col.div(rhs._col as Column<R>, memoryResource));
  }

  /**
   * True-divide this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to true-divide this Series by.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  true_div(rhs: bigint, memoryResource?: MemoryResource): Int64Series;
  true_div(rhs: number, memoryResource?: MemoryResource): Float64Series;
  true_div<R extends Numeric>(rhs: Scalar<R>,
                              memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  true_div<R extends Numeric>(rhs: NumericSeries<R>,
                              memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  true_div<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>,
                              memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.true_div(rhs, memoryResource));
      case 'number': return Series.new(this._col.true_div(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar
             ? Series.new(this._col.true_div(rhs, memoryResource))
             : Series.new(this._col.true_div(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Floor-divide this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to floor-divide this Series by.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  floor_div(rhs: bigint, memoryResource?: MemoryResource): Int64Series;
  floor_div(rhs: number, memoryResource?: MemoryResource): Float64Series;
  floor_div<R extends Numeric>(rhs: Scalar<R>,
                               memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  floor_div<R extends Numeric>(rhs: NumericSeries<R>,
                               memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  floor_div<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>,
                               memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.floor_div(rhs, memoryResource));
      case 'number': return Series.new(this._col.floor_div(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar
             ? Series.new(this._col.floor_div(rhs, memoryResource))
             : Series.new(this._col.floor_div(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Modulo this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to mod with this Series.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  mod(rhs: bigint, memoryResource?: MemoryResource): Int64Series;
  mod(rhs: number, memoryResource?: MemoryResource): Float64Series;
  mod<R extends Numeric>(rhs: Scalar<R>, memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  mod<R extends Numeric>(rhs: NumericSeries<R>,
                         memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  mod<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>, memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.mod(rhs, memoryResource));
      case 'number': return Series.new(this._col.mod(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar ? Series.new(this._col.mod(rhs, memoryResource))
                                 : Series.new(this._col.mod(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Power this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to use as the exponent for the power operation.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  pow(rhs: bigint, memoryResource?: MemoryResource): Int64Series;
  pow(rhs: number, memoryResource?: MemoryResource): Float64Series;
  pow<R extends Numeric>(rhs: Scalar<R>, memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  pow<R extends Numeric>(rhs: NumericSeries<R>,
                         memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  pow<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>, memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.pow(rhs, memoryResource));
      case 'number': return Series.new(this._col.pow(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar ? Series.new(this._col.pow(rhs, memoryResource))
                                 : Series.new(this._col.pow(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Perform the binary '==' operation between this column and another Series or scalar value.
   *
   * @rhs The other Series or scalar to compare with this column.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Series of booleans with the comparison result.
   */
  eq(rhs: bigint, memoryResource?: MemoryResource): Series<Bool8>;
  eq(rhs: number, memoryResource?: MemoryResource): Series<Bool8>;
  eq<R extends Numeric>(rhs: Scalar<R>, memoryResource?: MemoryResource): Series<Bool8>;
  eq<R extends Numeric>(rhs: NumericSeries<R>, memoryResource?: MemoryResource): Series<Bool8>;
  eq<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>, memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.eq(rhs, memoryResource));
      case 'number': return Series.new(this._col.eq(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar ? Series.new(this._col.eq(rhs, memoryResource))
                                 : Series.new(this._col.eq(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Perform the binary '!=' operation between this column and another Series or scalar value.
   *
   * @rhs The other Series or scalar to compare with this column.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Series of booleans with the comparison result.
   */
  ne(rhs: bigint, memoryResource?: MemoryResource): Series<Bool8>;
  ne(rhs: number, memoryResource?: MemoryResource): Series<Bool8>;
  ne<R extends Numeric>(rhs: Scalar<R>, memoryResource?: MemoryResource): Series<Bool8>;
  ne<R extends Numeric>(rhs: NumericSeries<R>, memoryResource?: MemoryResource): Series<Bool8>;
  ne<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>, memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.ne(rhs, memoryResource));
      case 'number': return Series.new(this._col.ne(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar ? Series.new(this._col.ne(rhs, memoryResource))
                                 : Series.new(this._col.ne(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Perform the binary '<' operation between this column and another Series or scalar value.
   *
   * @rhs The other Series or scalar to compare with this column.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Series of booleans with the comparison result.
   */
  lt(rhs: bigint, memoryResource?: MemoryResource): Series<Bool8>;
  lt(rhs: number, memoryResource?: MemoryResource): Series<Bool8>;
  lt<R extends Numeric>(rhs: Scalar<R>, memoryResource?: MemoryResource): Series<Bool8>;
  lt<R extends Numeric>(rhs: NumericSeries<R>, memoryResource?: MemoryResource): Series<Bool8>;
  lt<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>, memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.lt(rhs, memoryResource));
      case 'number': return Series.new(this._col.lt(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar ? Series.new(this._col.lt(rhs, memoryResource))
                                 : Series.new(this._col.lt(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Perform the binary '<=' operation between this column and another Series or scalar value.
   *
   * @rhs The other Series or scalar to compare with this column.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Series of booleans with the comparison result.
   */
  le(rhs: bigint, memoryResource?: MemoryResource): Series<Bool8>;
  le(rhs: number, memoryResource?: MemoryResource): Series<Bool8>;
  le<R extends Numeric>(rhs: Scalar<R>, memoryResource?: MemoryResource): Series<Bool8>;
  le<R extends Numeric>(rhs: NumericSeries<R>, memoryResource?: MemoryResource): Series<Bool8>;
  le<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>, memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.le(rhs, memoryResource));
      case 'number': return Series.new(this._col.le(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar ? Series.new(this._col.le(rhs, memoryResource))
                                 : Series.new(this._col.le(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Perform the binary '>' operation between this column and another Series or scalar value.
   *
   * @rhs The other Series or scalar to compare with this column.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Series of booleans with the comparison result.
   */
  gt(rhs: bigint, memoryResource?: MemoryResource): Series<Bool8>;
  gt(rhs: number, memoryResource?: MemoryResource): Series<Bool8>;
  gt<R extends Numeric>(rhs: Scalar<R>, memoryResource?: MemoryResource): Series<Bool8>;
  gt<R extends Numeric>(rhs: NumericSeries<R>, memoryResource?: MemoryResource): Series<Bool8>;
  gt<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>, memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.gt(rhs, memoryResource));
      case 'number': return Series.new(this._col.gt(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar ? Series.new(this._col.gt(rhs, memoryResource))
                                 : Series.new(this._col.gt(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Perform the binary '>=' operation between this column and another Series or scalar value.
   *
   * @rhs The other Series or scalar to compare with this column.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Series of booleans with the comparison result.
   */
  ge(rhs: bigint, memoryResource?: MemoryResource): Series<Bool8>;
  ge(rhs: number, memoryResource?: MemoryResource): Series<Bool8>;
  ge<R extends Numeric>(rhs: Scalar<R>, memoryResource?: MemoryResource): Series<Bool8>;
  ge<R extends Numeric>(rhs: NumericSeries<R>, memoryResource?: MemoryResource): Series<Bool8>;
  ge<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>, memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.ge(rhs, memoryResource));
      case 'number': return Series.new(this._col.ge(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar ? Series.new(this._col.ge(rhs, memoryResource))
                                 : Series.new(this._col.ge(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Perform a binary `&&` operation between this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to use.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  logical_and(rhs: bigint, memoryResource?: MemoryResource): Int64Series;
  logical_and(rhs: number, memoryResource?: MemoryResource): Float64Series;
  logical_and<R extends Numeric>(rhs: Scalar<R>,
                                 memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  logical_and<R extends Numeric>(rhs: NumericSeries<R>,
                                 memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  logical_and<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>,
                                 memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.logical_and(rhs, memoryResource));
      case 'number': return Series.new(this._col.logical_and(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar
             ? Series.new(this._col.logical_and(rhs, memoryResource))
             : Series.new(this._col.logical_and(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Perform a binary `||` operation between this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to use.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  logical_or(rhs: bigint, memoryResource?: MemoryResource): Int64Series;
  logical_or(rhs: number, memoryResource?: MemoryResource): Float64Series;
  logical_or<R extends Numeric>(rhs: Scalar<R>,
                                memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  logical_or<R extends Numeric>(rhs: NumericSeries<R>,
                                memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  logical_or<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>,
                                memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.logical_or(rhs, memoryResource));
      case 'number': return Series.new(this._col.logical_or(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar
             ? Series.new(this._col.logical_or(rhs, memoryResource))
             : Series.new(this._col.logical_or(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Perform a binary `coalesce` operation between this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to use.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  coalesce(rhs: bigint, memoryResource?: MemoryResource): Int64Series;
  coalesce(rhs: number, memoryResource?: MemoryResource): Float64Series;
  coalesce<R extends Numeric>(rhs: Scalar<R>,
                              memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  coalesce<R extends Numeric>(rhs: NumericSeries<R>,
                              memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  coalesce<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>,
                              memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.coalesce(rhs, memoryResource));
      case 'number': return Series.new(this._col.coalesce(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar
             ? Series.new(this._col.coalesce(rhs, memoryResource))
             : Series.new(this._col.coalesce(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Perform a binary `log_base` operation between this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to use.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  log_base(rhs: bigint, memoryResource?: MemoryResource): Int64Series;
  log_base(rhs: number, memoryResource?: MemoryResource): Float64Series;
  log_base<R extends Numeric>(rhs: Scalar<R>,
                              memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  log_base<R extends Numeric>(rhs: NumericSeries<R>,
                              memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  log_base<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>,
                              memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.log_base(rhs, memoryResource));
      case 'number': return Series.new(this._col.log_base(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar
             ? Series.new(this._col.log_base(rhs, memoryResource))
             : Series.new(this._col.log_base(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Perform a binary `atan2` operation between this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to use.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  atan2(rhs: bigint, memoryResource?: MemoryResource): Int64Series;
  atan2(rhs: number, memoryResource?: MemoryResource): Float64Series;
  atan2<R extends Numeric>(rhs: Scalar<R>,
                           memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  atan2<R extends Numeric>(rhs: NumericSeries<R>,
                           memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  atan2<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>,
                           memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.atan2(rhs, memoryResource));
      case 'number': return Series.new(this._col.atan2(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar
             ? Series.new(this._col.atan2(rhs, memoryResource))
             : Series.new(this._col.atan2(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Perform a binary `null_equals` operation between this Series and another Series or scalar
   * value.
   *
   * @param rhs The other Series or scalar to use.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  null_equals(rhs: bigint, memoryResource?: MemoryResource): Series<Bool8>;
  null_equals(rhs: number, memoryResource?: MemoryResource): Series<Bool8>;
  null_equals<R extends Numeric>(rhs: Scalar<R>, memoryResource?: MemoryResource): Series<Bool8>;
  null_equals<R extends Numeric>(rhs: NumericSeries<R>,
                                 memoryResource?: MemoryResource): Series<Bool8>;
  null_equals<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>,
                                 memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.null_equals(rhs, memoryResource));
      case 'number': return Series.new(this._col.null_equals(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar
             ? Series.new(this._col.null_equals(rhs, memoryResource))
             : Series.new(this._col.null_equals(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Perform a binary `null_max` operation between this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to use.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  null_max(rhs: bigint, memoryResource?: MemoryResource): Int64Series;
  null_max(rhs: number, memoryResource?: MemoryResource): Float64Series;
  null_max<R extends Numeric>(rhs: Scalar<R>,
                              memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  null_max<R extends Numeric>(rhs: NumericSeries<R>,
                              memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  null_max<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>,
                              memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.null_max(rhs, memoryResource));
      case 'number': return Series.new(this._col.null_max(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar
             ? Series.new(this._col.null_max(rhs, memoryResource))
             : Series.new(this._col.null_max(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Perform a binary `null_min` operation between this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to use.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  null_min(rhs: bigint, memoryResource?: MemoryResource): Int64Series;
  null_min(rhs: number, memoryResource?: MemoryResource): Float64Series;
  null_min<R extends Numeric>(rhs: Scalar<R>,
                              memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  null_min<R extends Numeric>(rhs: NumericSeries<R>,
                              memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  null_min<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>,
                              memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.null_min(rhs, memoryResource));
      case 'number': return Series.new(this._col.null_min(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar
             ? Series.new(this._col.null_min(rhs, memoryResource))
             : Series.new(this._col.null_min(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Compute the trigonometric sine for each value in this Series.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns A Series of the same number of elements containing the result of the operation.
   */
  sin(memoryResource?: MemoryResource): Series<T> {
    return Series.new(this._col.sin(memoryResource));
  }

  /**
   * Compute the trigonometric cosine for each value in this Series.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns A Series of the same number of elements containing the result of the operation.
   */
  cos(memoryResource?: MemoryResource): Series<T> {
    return Series.new(this._col.cos(memoryResource));
  }

  /**
   * Compute the trigonometric tangent for each value in this Series.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns A Series of the same number of elements containing the result of the operation.
   */
  tan(memoryResource?: MemoryResource): Series<T> {
    return Series.new(this._col.tan(memoryResource));
  }

  /**
   * Compute the trigonometric sine inverse for each value in this Series.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns A Series of the same number of elements containing the result of the operation.
   */
  asin(memoryResource?: MemoryResource): Series<T> {
    return Series.new(this._col.asin(memoryResource));
  }

  /**
   * Compute the trigonometric cosine inverse for each value in this Series.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns A Series of the same number of elements containing the result of the operation.
   */
  acos(memoryResource?: MemoryResource): Series<T> {
    return Series.new(this._col.acos(memoryResource));
  }

  /**
   * Compute the trigonometric tangent inverse for each value in this Series.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns A Series of the same number of elements containing the result of the operation.
   */
  atan(memoryResource?: MemoryResource): Series<T> {
    return Series.new(this._col.atan(memoryResource));
  }

  /**
   * Compute the hyperbolic sine for each value in this Series.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns A Series of the same number of elements containing the result of the operation.
   */
  sinh(memoryResource?: MemoryResource): Series<T> {
    return Series.new(this._col.sinh(memoryResource));
  }

  /**
   * Compute the hyperbolic cosine for each value in this Series.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns A Series of the same number of elements containing the result of the operation.
   */
  cosh(memoryResource?: MemoryResource): Series<T> {
    return Series.new(this._col.cosh(memoryResource));
  }

  /**
   * Compute the hyperbolic tangent for each value in this Series.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns A Series of the same number of elements containing the result of the operation.
   */
  tanh(memoryResource?: MemoryResource): Series<T> {
    return Series.new(this._col.tanh(memoryResource));
  }

  /**
   * Compute the hyperbolic sine inverse for each value in this Series.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns A Series of the same number of elements containing the result of the operation.
   */
  asinh(memoryResource?: MemoryResource): Series<T> {
    return Series.new(this._col.asinh(memoryResource));
  }

  /**
   * Compute the hyperbolic cosine inverse for each value in this Series.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns A Series of the same number of elements containing the result of the operation.
   */
  acosh(memoryResource?: MemoryResource): Series<T> {
    return Series.new(this._col.acosh(memoryResource));
  }

  /**
   * Compute the hyperbolic tangent inverse for each value in this Series.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns A Series of the same number of elements containing the result of the operation.
   */
  atanh(memoryResource?: MemoryResource): Series<T> {
    return Series.new(this._col.atanh(memoryResource));
  }

  /**
   * Compute the exponential (base e, euler number) for each value in this Series.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns A Series of the same number of elements containing the result of the operation.
   */
  exp(memoryResource?: MemoryResource): Series<T> {
    return Series.new(this._col.exp(memoryResource));
  }

  /**
   * Compute the natural logarithm (base e) for each value in this Series.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns A Series of the same number of elements containing the result of the operation.
   */
  log(memoryResource?: MemoryResource): Series<T> {
    return Series.new(this._col.log(memoryResource));
  }

  /**
   * Compute the square-root (x^0.5) for each value in this Series.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns A Series of the same number of elements containing the result of the operation.
   */
  sqrt(memoryResource?: MemoryResource): Series<T> {
    return Series.new(this._col.sqrt(memoryResource));
  }

  /**
   * Compute the cube-root (x^(1.0/3)) for each value in this Series.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns A Series of the same number of elements containing the result of the operation.
   */
  cbrt(memoryResource?: MemoryResource): Series<T> {
    return Series.new(this._col.cbrt(memoryResource));
  }

  /**
   * Compute the smallest integer value not less than arg for each value in this Series.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns A Series of the same number of elements containing the result of the operation.
   */
  ceil(memoryResource?: MemoryResource): Series<T> {
    return Series.new(this._col.ceil(memoryResource));
  }

  /**
   * Compute the largest integer value not greater than arg for each value in this Series.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns A Series of the same number of elements containing the result of the operation.
   */
  floor(memoryResource?: MemoryResource): Series<T> {
    return Series.new(this._col.floor(memoryResource));
  }

  /**
   * Compute the absolute value for each value in this Series.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns A Series of the same number of elements containing the result of the operation.
   */
  abs(memoryResource?: MemoryResource): Series<T> {
    return Series.new(this._col.abs(memoryResource));
  }

  /**
   * Compute the logical not (!) for each value in this Series.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns A Series of the same number of elements containing the result of the operation.
   */
  not(memoryResource?: MemoryResource): Series<Bool8> {
    return Series.new(this._col.not(memoryResource));
  }

  _process_reduction(skipna = true, memoryResource?: MemoryResource): Series<T> {
    return skipna ? this.dropNulls(memoryResource) : this.__construct(this._col);
  }

  /**
   * Compute the min of all values in this Column.
   * @param skipna The optional skipna if true drops NA and null values before computing reduction,
   * else if skipna is false, reduction is computed directly.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns The min of all the values in this Column.
   */
  min(skipna = true, memoryResource?: MemoryResource) {
    return this._process_reduction(skipna, memoryResource)._col.min(memoryResource);
  }

  /**
   * Compute the max of all values in this Column.
   * @param skipna The optional skipna if true drops NA and null values before computing reduction,
   * else if skipna is false, reduction is computed directly.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns The max of all the values in this Column.
   */
  max(skipna = true, memoryResource?: MemoryResource) {
    return this._process_reduction(skipna, memoryResource)._col.max(memoryResource);
  }

  /**
   * Compute a pair of [min,max] of all values in this Column.
   * @param skipna The optional skipna if true drops NA and null values before computing reduction,
   * else if skipna is false, reduction is computed directly.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns The pair of [min,max] of all the values in this Column.
   */
  minmax(skipna = true, memoryResource?: MemoryResource) {
    return this._process_reduction(skipna, memoryResource)._col.minmax(memoryResource);
  }

  /**
   * Compute the sum of all values in this Series.
   * @param skipna The optional skipna if true drops NA and null values before computing reduction,
   * else if skipna is false, reduction is computed directly.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns The sum of all the values in this Series.
   */
  sum(skipna = true, memoryResource?: MemoryResource) {
    return this._process_reduction(skipna, memoryResource)._col.sum(memoryResource);
  }

  /**
   * Compute the product of all values in this Series.
   *
   * @param skipna The optional skipna if true drops NA and null values before computing reduction,
   * else if skipna is false, reduction is computed directly.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns The product of all the values in this Series.
   */
  product(skipna = true, memoryResource?: MemoryResource) {
    return this._process_reduction(skipna, memoryResource)._col.product(memoryResource);
  }

  /**
   * Compute the sumOfSquares of all values in this Series.
   *
   * @param skipna The optional skipna if true drops NA and null values before computing reduction,
   * else if skipna is false, reduction is computed directly.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns The sumOfSquares of all the values in this Series.
   */
  sumOfSquares(skipna = true, memoryResource?: MemoryResource) {
    return this._process_reduction(skipna, memoryResource)._col.sum_of_squares(memoryResource);
  }

  /**
   * Compute the mean of all values in this Series.
   *
   * @param skipna The optional skipna if true drops NA and null values before computing reduction,
   * else if skipna is false, reduction is computed directly.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns The mean of all the values in this Series.
   */
  mean(skipna = true, memoryResource?: MemoryResource) {
    return this._process_reduction(skipna, memoryResource)._col.mean(memoryResource);
  }

  /**
   * Compute the median of all values in this Series.
   *
   * @param skipna The optional skipna if true drops NA and null values before computing reduction,
   * else if skipna is false, reduction is computed directly.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns The median of all the values in this Series.
   */
  median(skipna = true, memoryResource?: MemoryResource) {
    return this._process_reduction(skipna, memoryResource)._col.mean(memoryResource);
  }

  /**
   * Compute the nunique of all values in this Series.
   *
   * @param skipna The optional skipna if true drops NA and null values before computing reduction,
   * else if skipna is false, reduction is computed directly.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns The number of unqiue values in this Series.
   */
  nunique(skipna = true, memoryResource?: MemoryResource) {
    return this.nullCount === this.length ? skipna ? 0 : 1
                                          : this._col.nunique(skipna, memoryResource);
  }

  /**
   * Return unbiased variance of the Series.
   * Normalized by N-1 by default. This can be changed using the `ddof` argument
   *
   * @param skipna Exclude NA/null values. If an entire row/column is NA, the result will be NA.
   * @param ddof Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
   *  where N represents the number of elements.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns The unbiased variance of all the values in this Series.
   */
  var(skipna = true, ddof = 1, memoryResource?: MemoryResource) {
    return this._process_reduction(skipna, memoryResource)._col.var(ddof, memoryResource);
  }

  /**
   * Return sample standard deviation of the Series.
   * Normalized by N-1 by default. This can be changed using the `ddof` argument
   *
   * @param skipna Exclude NA/null values. If an entire row/column is NA, the result will be NA.
   * @param ddof Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
   *  where N represents the number of elements.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns The standard deviation of all the values in this Series.
   */
  std(skipna = true, ddof = 1, memoryResource?: MemoryResource) {
    return this._process_reduction(skipna, memoryResource)._col.std(ddof, memoryResource);
  }

  /**
   * Return values at the given quantile.
   *
   * @param q  the quantile(s) to compute, 0 <= q <= 1
   * @param interpolation This optional parameter specifies the interpolation method to use,
   *  when the desired quantile lies between two data points i and j.
   *  Valid values: ’linear’, ‘lower’, ‘higher’, ‘midpoint’, ‘nearest’.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns values at the given quantile.
   */
  quantile(q                                         = 0.5,
           interpolation: keyof typeof Interpolation = 'linear',
           memoryResource?: MemoryResource) {
    return this._process_reduction(true)._col.quantile(
      q, Interpolation[interpolation], memoryResource);
  }
}
