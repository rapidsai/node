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

import {MemoryResource} from '@rapidsai/rmm';
import {compareTypes} from 'apache-arrow/visitor/typecomparator';

import {Column} from '../column';
import {Scalar} from '../scalar';
import {Series} from '../series';
import {
  Bool8,
  Float32,
  Float64,
  Int16,
  Int32,
  Int64,
  Int8,
  Numeric,
  Uint16,
  Uint32,
  Uint64,
  Uint8
} from '../types/dtypes';
import {CommonType, Interpolation} from '../types/mappings';

import {Float64Series} from './float';
import {Int64Series} from './integral';

/**
 * A base class for Series of fixed-width numeric values.
 */
export abstract class NumericSeries<T extends Numeric> extends Series<T> {
  _castAsBool8(memoryResource?: MemoryResource): Series<Bool8> {  //
    return this._castNumeric(new Bool8, memoryResource);
  }
  _castAsInt8(memoryResource?: MemoryResource): Series<Int8> {  //
    return this._castNumeric(new Int8, memoryResource);
  }
  _castAsInt16(memoryResource?: MemoryResource): Series<Int16> {  //
    return this._castNumeric(new Int16, memoryResource);
  }
  _castAsInt32(memoryResource?: MemoryResource): Series<Int32> {  //
    return this._castNumeric(new Int32, memoryResource);
  }
  _castAsInt64(memoryResource?: MemoryResource): Series<Int64> {  //
    return this._castNumeric(new Int64, memoryResource);
  }
  _castAsUint8(memoryResource?: MemoryResource): Series<Uint8> {  //
    return this._castNumeric(new Uint8, memoryResource);
  }
  _castAsUint16(memoryResource?: MemoryResource): Series<Uint16> {  //
    return this._castNumeric(new Uint16, memoryResource);
  }
  _castAsUint32(memoryResource?: MemoryResource): Series<Uint32> {  //
    return this._castNumeric(new Uint32, memoryResource);
  }
  _castAsUint64(memoryResource?: MemoryResource): Series<Uint64> {  //
    return this._castNumeric(new Uint64, memoryResource);
  }
  _castAsFloat32(memoryResource?: MemoryResource): Series<Float32> {  //
    return this._castNumeric(new Float32, memoryResource);
  }
  _castAsFloat64(memoryResource?: MemoryResource): Series<Float64> {  //
    return this._castNumeric(new Float64, memoryResource);
  }

  protected _castNumeric<R extends Numeric>(type: R, memoryResource?: MemoryResource): Series<R> {
    return Series.new<R>(compareTypes(this.type, type) ? this._col as any as Column<R>
                                                       : this._col.cast(type, memoryResource));
  }

  /** @ignore */
  /* eslint-disable @typescript-eslint/no-unused-vars */
  nansToNulls(_memoryResource?: MemoryResource): Series<T> { return this.__construct(this._col); }
  /* eslint-enable @typescript-eslint/no-unused-vars */

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
   * Return a value at the specified index to host memory
   *
   * @param index the index in this Series to return a value for
   *
   * @example
   * ```typescript
   * import {Series} from "@rapidsai/cudf";
   *
   * // Float64Series
   * Series.new([1, 2, 3]).getValue(0) // 1
   * Series.new([1, 2, 3]).getValue(2) // 3
   * Series.new([1, 2, 3]).getValue(3) // throws index out of bounds error
   * ```
   */
  getValue(index: number) { return this._col.getValue(index); }

  /**
   * set value at the specified index
   *
   * @param index the index in this Series to set a value for
   * @param value the value to set at `index`
   *
   * @example
   * ```typescript
   * import {Series} from "@rapidsai/cudf";
   *
   * // Float64Series
   * const a = Series.new([1, 2, 3]);
   * a.setValue(0, -1) // inplace update -> Series([-1, 2, 3])
   * ```
   */
  setValue(index: number, value: T['scalarType']): void {
    this._col = this.scatter(value, [index])._col as Column<T>;
  }

  /**
   * Add this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to add to this Series.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Series of a common numeric type with the results of the binary operation.
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([10, 12, 14, 20]);
   * const b = Series.new([3, 2, 1, 3]);
   *
   * a.add(3); // [13, 15, 17, 23]
   * a.add(b); // [13, 14, 15, 23]
   * ```
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
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([10, 12, 14, 20]);
   * const b = Series.new([3, 2, 1, 3]);
   *
   * a.sub(3); // [7, 9, 11, 17]
   * a.sub(b); // [7, 10, 13, 17]
   * ```
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
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([10, 12, 14, 20]);
   * const b = Series.new([3, 2, 1, 3]);
   *
   * a.mul(3); // [30, 36, 42, 60]
   * a.mul(b); // [30, 24, 14, 60]
   * ```
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
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([10, 12, 14, 20]);
   * const b = Series.new([3, 2, 1, 3]);
   *
   * a.div(3); // [3.3333333333333335, 4, 4.666666666666667, 6.666666666666667]
   * a.div(b); // [3.3333333333333335, 6, 14, 6.666666666666667]
   * ```
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
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([10, 12, 14, 20]);
   * const b = Series.new([3, 2, 1, 3]);
   *
   * a.trueDiv(3); // [3.3333333333333335, 4, 4.666666666666667, 6.666666666666667]
   * a.trueDiv(b); // [3.3333333333333335, 6, 14, 6.666666666666667]
   * ```
   */
  trueDiv(rhs: bigint, memoryResource?: MemoryResource): Int64Series;
  trueDiv(rhs: number, memoryResource?: MemoryResource): Float64Series;
  trueDiv<R extends Numeric>(rhs: Scalar<R>,
                             memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  trueDiv<R extends Numeric>(rhs: NumericSeries<R>,
                             memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  trueDiv<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>,
                             memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.trueDiv(rhs, memoryResource));
      case 'number': return Series.new(this._col.trueDiv(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar
             ? Series.new(this._col.trueDiv(rhs, memoryResource))
             : Series.new(this._col.trueDiv(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Floor-divide this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to floor-divide this Series by.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Series of a common numeric type with the results of the binary operation.
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([10, 12, 14, 20]);
   * const b = Series.new([3, 2, 1, 3]);
   *
   * a.floorDiv(3); // [ 3, 4, 4, 6 ]
   * a.floorDiv(b); // [ 3, 6, 14, 6 ]
   * ```
   */
  floorDiv(rhs: bigint, memoryResource?: MemoryResource): Int64Series;
  floorDiv(rhs: number, memoryResource?: MemoryResource): Float64Series;
  floorDiv<R extends Numeric>(rhs: Scalar<R>,
                              memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  floorDiv<R extends Numeric>(rhs: NumericSeries<R>,
                              memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  floorDiv<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>,
                              memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.floorDiv(rhs, memoryResource));
      case 'number': return Series.new(this._col.floorDiv(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar
             ? Series.new(this._col.floorDiv(rhs, memoryResource))
             : Series.new(this._col.floorDiv(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Modulo this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to mod with this Series.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Series of a common numeric type with the results of the binary operation.
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([10, 12, 14, 20]);
   * const b = Series.new([3, 2, 1, 3]);
   *
   * a.mod(3); // [ 1, 0, 2, 2 ]
   * a.mod(b); // [ 1, 0, 0, 2 ]
   * ```
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
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([0, 1, 2, 3]);
   * const b = Series.new([3, 2, 1, 3]);
   *
   * a.pow(2); // [ 0, 1, 4, 9 ]
   * a.pow(b); // [ 0, 1, 2, 27 ]
   * ```
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
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([0, 1, 2, 3]);
   * const b = Series.new([3, 2, 1, 3]);
   *
   * a.eq(1); // [ false, true, false, false ]
   * a.eq(b); // [ false, false, false, true ]
   * ```
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
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([0, 1, 2, 3]);
   * const b = Series.new([3, 2, 1, 3]);
   *
   * a.ne(1); // [true, false, true, true]
   * a.ne(b); // [true, true, true, false]
   * ```
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
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([0, 1, 2, 3]);
   * const b = Series.new([3, 2, 1, 3]);
   *
   * a.lt(1); // [true, false, false, false]
   * a.lt(b); // [true, true, false, false]
   * ```
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
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([0, 1, 2, 3]);
   * const b = Series.new([3, 2, 1, 3]);
   *
   * a.le(1); // [true, true, false, false]
   * a.le(b); // [true, true, false, true]
   * ```
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
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([0, 1, 2, 3]);
   * const b = Series.new([3, 2, 1, 3]);
   *
   * a.gt(1); // [false, false, true, true]
   * a.gt(b); // [false, false, true, false]
   * ```
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
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([0, 1, 2, 3]);
   * const b = Series.new([3, 2, 1, 3]);
   *
   * a.ge(1); // [false, true, true, true]
   * a.ge(b); // [false, false, true, true]
   * ```
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
   * @example
   * ```typescript
   * import {Series, Bool8} from '@rapidsai/cudf';
   * const a = Series.new([false, true, true, false]);
   * const b = Series.new([false, false, false, false]);
   *
   * a.logicalAnd(0); // Float64Series [ 0, 0, 0, 0 ]
   * a.logicalAnd(0).view(new Bool8); // Bool8Series [ false, false, false, false ]
   * a.logicalAnd(b); // [false, false, false, false]
   * ```
   */
  logicalAnd(rhs: bigint, memoryResource?: MemoryResource): Int64Series;
  logicalAnd(rhs: number, memoryResource?: MemoryResource): Float64Series;
  logicalAnd<R extends Numeric>(rhs: Scalar<R>,
                                memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  logicalAnd<R extends Numeric>(rhs: NumericSeries<R>,
                                memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  logicalAnd<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>,
                                memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.logicalAnd(rhs, memoryResource));
      case 'number': return Series.new(this._col.logicalAnd(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar
             ? Series.new(this._col.logicalAnd(rhs, memoryResource))
             : Series.new(this._col.logicalAnd(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Perform a binary `||` operation between this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to use.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Series of a common numeric type with the results of the binary operation.
   * @example
   * ```typescript
   * import {Series, Bool8} from '@rapidsai/cudf';
   * const a = Series.new([false, true, true, false]);
   * const b = Series.new([false, false, false, false]);
   *
   * a.logicalOr(0); // Float64Series [ 0, 1, 1, 0 ]
   * a.logicalOr(0).cast(new Bool8); // Bool8Series [ false, true, true, false ]
   * a.logicalOr(b); // [false, true, true, false]
   * ```
   */
  logicalOr(rhs: bigint, memoryResource?: MemoryResource): Int64Series;
  logicalOr(rhs: number, memoryResource?: MemoryResource): Float64Series;
  logicalOr<R extends Numeric>(rhs: Scalar<R>,
                               memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  logicalOr<R extends Numeric>(rhs: NumericSeries<R>,
                               memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  logicalOr<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>,
                               memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.logicalOr(rhs, memoryResource));
      case 'number': return Series.new(this._col.logicalOr(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar
             ? Series.new(this._col.logicalOr(rhs, memoryResource))
             : Series.new(this._col.logicalOr(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Perform a binary `logBase` operation between this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to use.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Series of a common numeric type with the results of the binary operation.
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([1, 10, 100]);
   * const b = Series.new([2, 10, 20]);
   *
   * a.logBase(10); // [0, 1, 2]
   * a.logBase(b); // [0, 1, 1.537243573680482]
   * ```
   */
  logBase(rhs: bigint, memoryResource?: MemoryResource): Int64Series;
  logBase(rhs: number, memoryResource?: MemoryResource): Float64Series;
  logBase<R extends Numeric>(rhs: Scalar<R>,
                             memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  logBase<R extends Numeric>(rhs: NumericSeries<R>,
                             memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  logBase<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>,
                             memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.logBase(rhs, memoryResource));
      case 'number': return Series.new(this._col.logBase(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar
             ? Series.new(this._col.logBase(rhs, memoryResource))
             : Series.new(this._col.logBase(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Perform a binary `atan2` operation between this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to use.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Series of a common numeric type with the results of the binary operation.
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([1, 3, 5, null, 7]);
   * const b = Series.new([1, 3, 3, null, 9]);
   *
   * a.atan2(3);
   * // [0.3217505543966422, 0.7853981633974483, 1.0303768265243125, 0, 1.1659045405098132]
   *
   * a.atan2(b);
   * // [0.7853981633974483, 0.7853981633974483, 1.0303768265243125, 0, 0.6610431688506869]
   * ```
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
   * Perform a binary `nullEquals` operation between this Series and another Series or scalar
   * value.
   *
   * @param rhs The other Series or scalar to use.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Series of a common numeric type with the results of the binary operation.
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([1, 3, 5, null, 7]);
   * const b = Series.new([1, 3, 3, null, 9]);
   *
   * a.nullEquals(3); // [false, true, false, false, false]
   * a.nullEquals(b); // [true, true, false, true, false]
   * ```
   */
  nullEquals(rhs: bigint, memoryResource?: MemoryResource): Series<Bool8>;
  nullEquals(rhs: number, memoryResource?: MemoryResource): Series<Bool8>;
  nullEquals<R extends Numeric>(rhs: Scalar<R>, memoryResource?: MemoryResource): Series<Bool8>;
  nullEquals<R extends Numeric>(rhs: NumericSeries<R>,
                                memoryResource?: MemoryResource): Series<Bool8>;
  nullEquals<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>,
                                memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.nullEquals(rhs, memoryResource));
      case 'number': return Series.new(this._col.nullEquals(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar
             ? Series.new(this._col.nullEquals(rhs, memoryResource))
             : Series.new(this._col.nullEquals(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Perform a binary `nullMax` operation between this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to use.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Series of a common numeric type with the results of the binary operation.
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([1, 3, 5, null, 7]);
   * const b = Series.new([6, 6, 6, 6, 6]);
   *
   * a.nullMax(4); // [4, 4, 5, 4, 7]
   * a.nullMax(b); // [6, 6, 6, 6, 7]
   * ```
   */
  nullMax(rhs: bigint, memoryResource?: MemoryResource): Int64Series;
  nullMax(rhs: number, memoryResource?: MemoryResource): Float64Series;
  nullMax<R extends Numeric>(rhs: Scalar<R>,
                             memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  nullMax<R extends Numeric>(rhs: NumericSeries<R>,
                             memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  nullMax<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>,
                             memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.nullMax(rhs, memoryResource));
      case 'number': return Series.new(this._col.nullMax(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar
             ? Series.new(this._col.nullMax(rhs, memoryResource))
             : Series.new(this._col.nullMax(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Perform a binary `nullMin` operation between this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to use.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Series of a common numeric type with the results of the binary operation.
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([1, 3, 5, null, 7]);
   * const c = Series.new([6, 6, 6, 6, 6]);
   *
   * a.nullMin(4); // [1, 3, 4, 4, 4]
   * a.nullMin(b); // [1, 3, 5, 6, 6]
   * ```
   */
  nullMin(rhs: bigint, memoryResource?: MemoryResource): Int64Series;
  nullMin(rhs: number, memoryResource?: MemoryResource): Float64Series;
  nullMin<R extends Numeric>(rhs: Scalar<R>,
                             memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  nullMin<R extends Numeric>(rhs: NumericSeries<R>,
                             memoryResource?: MemoryResource): Series<CommonType<T, R>>;
  nullMin<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>,
                             memoryResource?: MemoryResource) {
    switch (typeof rhs) {
      case 'bigint': return Series.new(this._col.nullMin(rhs, memoryResource));
      case 'number': return Series.new(this._col.nullMin(rhs, memoryResource));
      default: break;
    }
    return rhs instanceof Scalar
             ? Series.new(this._col.nullMin(rhs, memoryResource))
             : Series.new(this._col.nullMin(rhs._col as Column<R>, memoryResource));
  }

  /**
   * Compute the trigonometric sine for each value in this Series.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns A Series of the same number of elements containing the result of the operation.
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * Series.new([0, 45, 1]).sin(); // [0, 0.8509035245341184, 0.8414709848078965]
   * ```
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
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * Series.new([0, 45, 1]).cos(); // [1, 0.5253219888177297, 0.5403023058681398]
   * ```
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
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * Series.new([0, 45, 1]).tan(); // [0, 1.6197751905438615, 1.557407724654902]
   * ```
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
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * Series.new([0, 45, 1]).asin(); // [0, NaN, 1.5707963267948966]
   * ```
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
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * Series.new([0, 45, 1]).acos(); // [1.5707963267948966, NaN, 0]
   * ```
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
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * Series.new([0, 45, 1]).atan(); // [0, 1.5485777614681775, 0.7853981633974483]
   * ```
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
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * Series.new([0, 45, 1]).sinh(); // [0, 17467135528742547000, 1.1752011936438014]
   * ```
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
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * Series.new([0, 45, 1]).cosh(); // [1, 17467135528742547000, 1.5430806348152437]
   * ```
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
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * Series.new([0, 45, 1]).tanh(); // [0, 1, 0.7615941559557649]
   * ```
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
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * Series.new([0, 45, 1]).asinh(); // [0, 4.49993310426429, 0.8813735870195429]
   * ```
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
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * Series.new([7, 56, 1]).acosh(); // [2.6339157938496336, 4.71841914237288, 0]
   * ```
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
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * Series.new([0, -0.5]).atanh(); // [0, -0.5493061443340549]
   * ```
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
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * Series.new([-1.2, 2.5]).exp(); // [0.30119421191220214, 12.182493960703473]
   * ```
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
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * Series.new([-1.2, 2.5, 4]).log(); // [NaN, 0.9162907318741551, 1.3862943611198906]
   * ```
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
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * Series.new([-1.2, 2.5, 4]).sqrt(); // [NaN, 1.5811388300841898, 2]
   * ```
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
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * Series.new([-1.2, 2.5]).cbrt(); // [-1.0626585691826111, 1.3572088082974534]
   * ```
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
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * Series.new([-1.2, 2.5, -3, 4.6, 5]).ceil(); // [-1, 3, -3, 5, 5]
   * ```
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
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * Series.new([-1.2, 2.5, -3, 4.6, 5]).floor(); // [-2, 2, -3, 4, 5]
   * ```
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
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * Series.new([-1, 2, -3, 4, 5]).abs(); // [1, 2, 3, 4, 5]
   * ```
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
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([true, false, true, true, false])
   * const b = Series.new([0, 1, 2, 3, 4])
   *
   * a.not() // [false, true, false, false, true]
   * b.not() // [true, false, false, false, false]
   */
  not(memoryResource?: MemoryResource): Series<Bool8> {
    return Series.new(this._col.not(memoryResource));
  }

  /**
   * Compute the min of all values in this Column.
   * @param skipNulls The optional skipNulls if true drops NA and null values before computing
   *   reduction,
   * else if skipNulls is false, reduction is computed directly.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns The min of all the values in this Column.
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([5, 4, 1, 1, 1])
   *
   * a.min() // [1]
   */
  min(skipNulls = true, memoryResource?: MemoryResource) {
    const data = skipNulls ? this.nansToNulls().dropNulls() : this;
    return data._col.min(memoryResource);
  }

  /**
   * Compute the max of all values in this Column.
   * @param skipNulls The optional skipNulls if true drops NA and null values before computing
   *   reduction,
   * else if skipNulls is false, reduction is computed directly.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns The max of all the values in this Column.
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([5, 4, 1, 1, 1])
   *
   * a.max() // 5
   */
  max(skipNulls = true, memoryResource?: MemoryResource) {
    const data = skipNulls ? this.nansToNulls().dropNulls() : this;
    return data._col.max(memoryResource);
  }

  /**
   * Compute a pair of [min,max] of all values in this Column.
   * @param skipNulls The optional skipNulls if true drops NA and null values before computing
   *   reduction,
   * else if skipNulls is false, reduction is computed directly.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns The pair of [min,max] of all the values in this Column.
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([5, 4, 1, 1, 1])
   *
   * a.minmax() // [1,5]
   */
  minmax(skipNulls = true, memoryResource?: MemoryResource) {
    const data = skipNulls ? this.nansToNulls().dropNulls() : this;
    return data._col.minmax(memoryResource);
  }

  /**
   * Compute the sum of all values in this Series.
   * @param skipNulls The optional skipNulls if true drops NA and null values before computing
   *   reduction,
   * else if skipNulls is false, reduction is computed directly.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns The sum of all the values in this Series.
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([5, 4, 1, 1, 1])
   *
   * a.sum() // 20
   * ```
   */
  sum(skipNulls = true, memoryResource?: MemoryResource) {
    const data = skipNulls ? this.nansToNulls().dropNulls() : this;
    return data._col.sum(memoryResource);
  }

  /**
   * Compute the product of all values in this Series.
   *
   * @param skipNulls The optional skipNulls if true drops NA and null values before computing
   *   reduction,
   * else if skipNulls is false, reduction is computed directly.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns The product of all the values in this Series.
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([5, 4, 1, 1, 1])
   *
   * a.product() // 20
   * ```
   */
  product(skipNulls = true, memoryResource?: MemoryResource) {
    const data = skipNulls ? this.nansToNulls().dropNulls() : this;
    return data._col.product(memoryResource);
  }

  /**
   * Compute the sumOfSquares of all values in this Series.
   *
   * @param skipNulls The optional skipNulls if true drops NA and null values before computing
   *   reduction,
   * else if skipNulls is false, reduction is computed directly.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns The sumOfSquares of all the values in this Series.
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([5, 4, 1, 1, 1])
   *
   * a.sumOfSquares() // 44
   * ```
   */
  sumOfSquares(skipNulls = true, memoryResource?: MemoryResource) {
    const data = skipNulls ? this.nansToNulls().dropNulls() : this;
    return data._col.sumOfSquares(memoryResource);
  }

  /**
   * Compute the mean of all values in this Series.
   *
   * @param skipNulls The optional skipNulls if true drops NA and null values before computing
   *   reduction,
   * else if skipNulls is false, reduction is computed directly.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns The mean of all the values in this Series.
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([5, 4, 1, 1, 1])
   *
   * a.mean() // 2.4
   * ```
   */
  mean(skipNulls = true, memoryResource?: MemoryResource) {
    if (!skipNulls && this.nullCount > 0) { return NaN; }
    const data = skipNulls ? this.nansToNulls().dropNulls() : this;
    return data._col.mean(memoryResource);
  }

  /**
   * Compute the median of all values in this Series.
   *
   * @param skipNulls The optional skipNulls if true drops NA and null values before computing
   *   reduction,
   * else if skipNulls is false, reduction is computed directly.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns The median of all the values in this Series.
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([5, 4, 1, 1, 1])
   *
   * a.median() // 1
   * ```
   */
  median(skipNulls = true, memoryResource?: MemoryResource) {
    if (!skipNulls && this.nullCount > 0) { return NaN; }
    const data = skipNulls ? this.nansToNulls().dropNulls() : this;
    return data._col.median(memoryResource);
  }

  /**
   * Compute the nunique of all values in this Series.
   *
   * @param dropna The optional dropna if true drops NA and null values before computing reduction,
   * else if dropna is false, reduction is computed directly.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns The number of unqiue values in this Series.
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([1, 2, 3, 4, 4, 5, null, null]);
   *
   * a.nunique() // 5
   * a.nunique(false) // 6
   * ```
   */
  nunique(dropna = true, memoryResource?: MemoryResource) {
    return this.nullCount === this.length ? dropna ? 0 : 1
                                          : this._col.nunique(dropna, memoryResource);
  }

  /**
   * Return unbiased variance of the Series.
   * Normalized by N-1 by default. This can be changed using the `ddof` argument
   *
   * @param skipNulls Exclude NA/null values. If an entire row/column is NA, the result will be NA.
   * @param ddof Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
   *  where N represents the number of elements.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns The unbiased variance of all the values in this Series.
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([1, 2, 3, 4, 5, null]);
   *
   * a.var() // 2.5
   * a.var(true, 2) // 3.333333333333332
   * a.var(true, 5) // NaN, ddof>=a.length results in NaN
   * ```
   */
  var(skipNulls = true, ddof = 1, memoryResource?: MemoryResource) {
    const data = skipNulls ? this.nansToNulls().dropNulls() : this;
    return data._col.var(ddof, memoryResource);
  }

  /**
   * Return Fisher’s unbiased kurtosis of a sample.
   * Kurtosis obtained using Fisher’s definition of kurtosis (kurtosis of normal == 0.0). Normalized
   * by N-1.
   *
   * @param skipNulls Exclude NA/null values. If an entire row/column is NA, the result will be NA.
   * @returns The unbiased kurtosis of all the values in this Series.
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([1, 2, 3, 4]);
   *
   * a.kurtosis() // -1.1999999999999904
   * ```
   */
  kurtosis(skipNulls = true) {
    if (this.length == 0 || (this.hasNulls && !skipNulls)) { return NaN; }

    const data = skipNulls ? this.nansToNulls().dropNulls() : this;

    const n = data.length;
    if (n < 4) { return NaN; }

    const V = data.var(skipNulls, 1);  // ddof = 1
    if (V == 0) { return 0; }

    const mu = data.mean(skipNulls);

    const m4 = (data.sub(mu).pow(4).sum(skipNulls) ) / (V ** 2);

    // This is modeled after the cudf kurtosis implementation, it would be
    // nice to be able to point to a reference for this specific formula
    const a = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3)) * m4;
    const b = ((n - 1) ** 2) / ((n - 2) * (n - 3));

    return a - 3 * b;
  }

  /**
   * Return unbiased Fisher-Pearson skew of a sample.
   *
   * @param skipNulls Exclude NA/null values. If an entire row/column is NA, the result will be NA.
   * @returns The unbiased skew of all the values in this Series.
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([1, 2, 3, 4, 5, 6, 6]);
   *
   * a.skew() // -0.288195490292614
   * ```
   */
  skew(skipNulls = true) {
    if (this.length == 0 || (this.hasNulls && !skipNulls)) { return NaN; }

    const data = skipNulls ? this.nansToNulls().dropNulls() : this;

    const n = data.length;
    if (data.length < 3) { return NaN; }

    const V = data.var(skipNulls, 0);  // ddof = 0
    if (V == 0) { return 0; }

    const mu = data.mean(skipNulls);

    const m3 = (data.sub(mu).pow(3).sum(skipNulls) ) / n;

    return ((n * (n - 1)) ** 0.5) / (n - 2) * m3 / (V ** (3 / 2));
  }

  /**
   * Return sample standard deviation of the Series.
   * Normalized by N-1 by default. This can be changed using the `ddof` argument
   *
   * @param skipNulls Exclude NA/null values. If an entire row/column is NA, the result will be NA.
   * @param ddof Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
   *  where N represents the number of elements.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns The standard deviation of all the values in this Series.
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * const a = Series.new([1, 2, 3, 4, 5]);
   *
   * //skipNulls=true, ddof=1
   * a.std() // 1.5811388300841898
   * a.std(true, 2) // 1.8257418583505534
   * a.std(true, 5) // NaN, ddof>=a.length results in NaN
   * ```
   */
  std(skipNulls = true, ddof = 1, memoryResource?: MemoryResource) {
    const data = skipNulls ? this.nansToNulls().dropNulls() : this;
    return data._col.std(ddof, memoryResource);
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
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * const a = Series.new([1, 2, 3, 4, 5])
   *
   * a.quantile(0.3, "linear") // 2.2
   * a.quantile(0.3, "lower") // 2
   * a.quantile(0.3, "higher") // 3
   * a.quantile(0.3, "midpoint") // 2.5
   * a.quantile(0.3, "nearest") // 2
   * ```
   */
  quantile(q                                         = 0.5,
           interpolation: keyof typeof Interpolation = 'linear',
           memoryResource?: MemoryResource) {
    return this.nansToNulls().dropNulls()._col.quantile(
      q, Interpolation[interpolation], memoryResource);
  }

  /**
   * Return whether all elements are true in Series.
   *
   * @param skipNulls bool
   * Exclude null values. If the entire row/column is NA and skipNulls is true, then the result will
   * be true, as for an empty row/column. If skipNulls is false, then NA are treated as true,
   * because these are not equal to zero.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   *
   * @returns true if all elements are true in Series, else false.
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * //boolean series
   * Series.new([true, false, true]).all() // false
   * Series.new([true, true, true]).all() // true
   * ```
   */
  all(skipNulls = true, memoryResource?: MemoryResource) {
    if (skipNulls) {
      if (this.length == this.nullCount) { return true; }
    }
    return this._col.all(memoryResource);
  }

  /**
   * Return whether any elements are true in Series.
   *
   * @param skipNulls bool
   * Exclude NA/null values. If the entire row/column is NA and skipNulls is true, then the result
   * will be true, as for an empty row/column. If skipNulls is false, then NA are treated as true,
   * because these are not equal to zero.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   *
   * @returns true if any elements are true in Series, else false.
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * //boolean series
   * Series.new([false, false, false]).any() // false
   * Series.new([true, false, true]).any() // true
   * ```
   */
  any(skipNulls = true, memoryResource?: MemoryResource) {
    if (this.length == 0) { return false; }
    if (skipNulls) {
      if (this.length == this.nullCount) { return false; }
    }
    return this._col.any(memoryResource);
  }
}
