// Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

import {MemoryData} from '@nvidia/cuda';
import {DeviceBuffer} from '@nvidia/rmm';
import {RecordBatchReader, Table as ArrowTable, Vector} from 'apache-arrow';
import {VectorType} from 'apache-arrow/interfaces';

import {Column, ColumnProps} from './column';
import {fromArrow} from './column/from_arrow';
import {DataFrame} from './data_frame';
import {Scalar} from './scalar';
import {Table} from './table';
import {
  Bool8,
  CommonType,
  CUDFToArrowType,
  DataType,
  Float64,
  Int64,
  Integral,
  NullOrder,
  Numeric
} from './types';

export interface Series {
  getChild(index: number): Series;
  setNullCount(nullCount: number): void;
  setNullMask(mask: DeviceBuffer, nullCount?: number): void;
}

export type SeriesProps<T extends DataType = any> = {
  type: T,
  data?: DeviceBuffer|MemoryData|null,
  offset?: number,
  length?: number,
  nullCount?: number,
  nullMask?: DeviceBuffer|MemoryData|null,
  children?: ReadonlyArray<Series>|null
};

/**
 * One-dimensional GPU array
 */
export class Series<T extends DataType = any> {
  /*private*/ _data: Column<T>;

  constructor(value: SeriesProps<T>|Column<T>|Vector<CUDFToArrowType<T>>) {
    if (value instanceof Column) {
      this._data = value;
    } else if (value instanceof Vector) {
      this._data = fromArrow(value);
    } else {
      const props: ColumnProps = {
        type: value.type.id,
        data: value.data,
        offset: value.offset,
        length: value.length,
        nullCount: value.nullCount,
        nullMask: value.nullMask,
      };
      if (value.children != null) {
        props.children = value.children.map((item: Series) => item._data);
      }
      this._data = new Column<T>(props);
    }
  }

  /**
   * The data type of elements in the underlying data.
   */
  get type() { return this._data.type; }

  /**
   * The GPU buffer for the null-mask
   */
  get mask() { return this._data.mask; }

  /**
   * The number of elements in the underlying data.
   */
  get length() { return this._data.length; }

  /**
   * Whether a null-mask is needed
   */
  get nullable() { return this._data.nullable; }

  /**
   * Whether the Series contains null values.
   */
  get hasNulls() { return this._data.hasNulls; }

  /**
   * Number of null values
   */
  get nullCount() { return this._data.nullCount; }

  /**
   * The number of child columns
   */
  get numChildren() { return this._data.numChildren; }

  /**
   * Return a seb-selection of the Series from the specified indices
   *
   * @param selection
   */
  gather<R extends Integral>(selection: Series<R>) {
    return new Series<T>(this._data.gather(selection._data));
  }

  /**
   * Return a seb-selection of the Series from the specified boolean mask
   *
   * @param mask
   */
  filter(mask: Series<Bool8>) { return new Series<T>(this._data.gather(mask._data)); }

  /**
   * Return a child at the specified index to host memory
   *
   * @param index
   */
  getChild(index: number) { return new Series(this._data.getChild(index)); }

  /**
   * Return a value at the specified index to host memory
   *
   * @param index
   */
  getValue(index: number) { return this._data.getValue(index); }

  // setValue(index: number, value?: this[0] | null);

  /**
   * Copy the underlying device memory to host, and return an Iterator of the values.
   */
  [Symbol.iterator]() { return this.toArrow()[Symbol.iterator](); }

  /**
   *
   * @param mask The null-mask. Valid values are marked as 1; otherwise 0. The
   * mask bit given the data index idx is computed as:
   * ```
   * (mask[idx // 8] >> (idx % 8)) & 1
   * ```
   * @param nullCount The number of null values. If None, it is calculated
   * automatically.
   */
  setNullMask(mask: DeviceBuffer, nullCount?: number) { this._data.setNullMask(mask, nullCount); }

  /**
   * Copy a column to an Arrow vector in host memory
   */
  toArrow(): VectorType<CUDFToArrowType<T>> {
    const reader = RecordBatchReader.from(new Table({columns: [this._data]}).toArrow([[0]]));
    const column = new ArrowTable(reader.schema, [...reader]).getColumnAt<CUDFToArrowType<T>>(0);
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    return column!.chunks[0] as VectorType<CUDFToArrowType<T>>;
  }

  /**
   * Generate an ordering that sorts the Series in a specified way
   *
   * @param ascending whether to sort ascending (true) or descending (false)
   * @param null_order whether nulls should sort before or after other values
   *
   * @returns Series containting the permutation indices for the desired sort order
   */
  orderBy(ascending = true, null_order: NullOrder = NullOrder.BEFORE) {
    return new DataFrame({"col": this}).orderBy({"col": {ascending, null_order}});
  }

  /**
   * Generate a new Series that is sorted in a specified way
   *
   * @param ascending whether to sort ascending (true) or descending (false)
   *   Default: true
   * @param null_order whether nulls should sort before or after other values
   *   Default: BEFORE
   *
   * @returns Sorted values
   */
  sortValues(ascending = true, null_order: NullOrder = NullOrder.BEFORE) {
    return this.gather(this.orderBy(ascending, null_order));
  }

  /**
   * Add this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to add to this Series.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  add(rhs: bigint): Series<Int64>;
  add(rhs: number): Series<Float64>;
  add<R extends Numeric>(rhs: Scalar<R>): Series<CommonType<T, R>>;
  add<R extends Numeric>(rhs: Series<R>): Series<CommonType<T, R>>;
  add<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>) {
    switch (typeof rhs) {
      case 'bigint': return new Series(this._data.add(rhs));
      case 'number': return new Series(this._data.add(rhs));
      default: break;
    }
    return rhs instanceof Scalar ? new Series(this._data.add(rhs))
                                 : new Series(this._data.add(rhs._data));
  }

  /**
   * Subtract this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to subtract from this Series.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  sub(rhs: bigint): Series<Int64>;
  sub(rhs: number): Series<Float64>;
  sub<R extends Numeric>(rhs: Scalar<R>): Series<CommonType<T, R>>;
  sub<R extends Numeric>(rhs: Series<R>): Series<CommonType<T, R>>;
  sub<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>) {
    switch (typeof rhs) {
      case 'bigint': return new Series(this._data.sub(rhs));
      case 'number': return new Series(this._data.sub(rhs));
      default: break;
    }
    return rhs instanceof Scalar ? new Series(this._data.sub(rhs))
                                 : new Series(this._data.sub(rhs._data));
  }

  /**
   * Multiply this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to multiply this column by.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  mul(rhs: bigint): Series<Int64>;
  mul(rhs: number): Series<Float64>;
  mul<R extends Numeric>(rhs: Scalar<R>): Series<CommonType<T, R>>;
  mul<R extends Numeric>(rhs: Series<R>): Series<CommonType<T, R>>;
  mul<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>) {
    switch (typeof rhs) {
      case 'bigint': return new Series(this._data.mul(rhs));
      case 'number': return new Series(this._data.mul(rhs));
      default: break;
    }
    return rhs instanceof Scalar ? new Series(this._data.mul(rhs))
                                 : new Series(this._data.mul(rhs._data));
  }

  /**
   * Divide this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to divide this Series by.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  div(rhs: bigint): Series<Int64>;
  div(rhs: number): Series<Float64>;
  div<R extends Numeric>(rhs: Scalar<R>): Series<CommonType<T, R>>;
  div<R extends Numeric>(rhs: Series<R>): Series<CommonType<T, R>>;
  div<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>) {
    switch (typeof rhs) {
      case 'bigint': return new Series(this._data.div(rhs));
      case 'number': return new Series(this._data.div(rhs));
      default: break;
    }
    return rhs instanceof Scalar ? new Series(this._data.div(rhs))
                                 : new Series(this._data.div(rhs._data));
  }

  /**
   * True-divide this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to true-divide this Series by.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  true_div(rhs: bigint): Series<Int64>;
  true_div(rhs: number): Series<Float64>;
  true_div<R extends Numeric>(rhs: Scalar<R>): Series<CommonType<T, R>>;
  true_div<R extends Numeric>(rhs: Series<R>): Series<CommonType<T, R>>;
  true_div<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>) {
    switch (typeof rhs) {
      case 'bigint': return new Series(this._data.true_div(rhs));
      case 'number': return new Series(this._data.true_div(rhs));
      default: break;
    }
    return rhs instanceof Scalar ? new Series(this._data.true_div(rhs))
                                 : new Series(this._data.true_div(rhs._data));
  }

  /**
   * Floor-divide this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to floor-divide this Series by.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  floor_div(rhs: bigint): Series<Int64>;
  floor_div(rhs: number): Series<Float64>;
  floor_div<R extends Numeric>(rhs: Scalar<R>): Series<CommonType<T, R>>;
  floor_div<R extends Numeric>(rhs: Series<R>): Series<CommonType<T, R>>;
  floor_div<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>) {
    switch (typeof rhs) {
      case 'bigint': return new Series(this._data.floor_div(rhs));
      case 'number': return new Series(this._data.floor_div(rhs));
      default: break;
    }
    return rhs instanceof Scalar ? new Series(this._data.floor_div(rhs))
                                 : new Series(this._data.floor_div(rhs._data));
  }

  /**
   * Modulo this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to mod with this Series.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  mod(rhs: bigint): Series<Int64>;
  mod(rhs: number): Series<Float64>;
  mod<R extends Numeric>(rhs: Scalar<R>): Series<CommonType<T, R>>;
  mod<R extends Numeric>(rhs: Series<R>): Series<CommonType<T, R>>;
  mod<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>) {
    switch (typeof rhs) {
      case 'bigint': return new Series(this._data.mod(rhs));
      case 'number': return new Series(this._data.mod(rhs));
      default: break;
    }
    return rhs instanceof Scalar ? new Series(this._data.mod(rhs))
                                 : new Series(this._data.mod(rhs._data));
  }

  /**
   * Power this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to use as the exponent for the power operation.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  pow(rhs: bigint): Series<Int64>;
  pow(rhs: number): Series<Float64>;
  pow<R extends Numeric>(rhs: Scalar<R>): Series<CommonType<T, R>>;
  pow<R extends Numeric>(rhs: Series<R>): Series<CommonType<T, R>>;
  pow<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>) {
    switch (typeof rhs) {
      case 'bigint': return new Series(this._data.pow(rhs));
      case 'number': return new Series(this._data.pow(rhs));
      default: break;
    }
    return rhs instanceof Scalar ? new Series(this._data.pow(rhs))
                                 : new Series(this._data.pow(rhs._data));
  }

  /**
   * Perform the binary '==' operation between this column and another Series or scalar value.
   *
   * @rhs The other Series or scalar to compare with this column.
   * @returns A Series of booleans with the comparison result.
   */
  eq(rhs: bigint): Series<Bool8>;
  eq(rhs: number): Series<Bool8>;
  eq<R extends Numeric>(rhs: Scalar<R>): Series<Bool8>;
  eq<R extends Numeric>(rhs: Series<R>): Series<Bool8>;
  eq<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>) {
    switch (typeof rhs) {
      case 'bigint': return new Series(this._data.eq(rhs));
      case 'number': return new Series(this._data.eq(rhs));
      default: break;
    }
    return rhs instanceof Scalar ? new Series(this._data.eq(rhs))
                                 : new Series(this._data.eq(rhs._data));
  }

  /**
   * Perform the binary '!=' operation between this column and another Series or scalar value.
   *
   * @rhs The other Series or scalar to compare with this column.
   * @returns A Series of booleans with the comparison result.
   */
  ne(rhs: bigint): Series<Bool8>;
  ne(rhs: number): Series<Bool8>;
  ne<R extends Numeric>(rhs: Scalar<R>): Series<Bool8>;
  ne<R extends Numeric>(rhs: Series<R>): Series<Bool8>;
  ne<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>) {
    switch (typeof rhs) {
      case 'bigint': return new Series(this._data.ne(rhs));
      case 'number': return new Series(this._data.ne(rhs));
      default: break;
    }
    return rhs instanceof Scalar ? new Series(this._data.ne(rhs))
                                 : new Series(this._data.ne(rhs._data));
  }

  /**
   * Perform the binary '<' operation between this column and another Series or scalar value.
   *
   * @rhs The other Series or scalar to compare with this column.
   * @returns A Series of booleans with the comparison result.
   */
  lt(rhs: bigint): Series<Bool8>;
  lt(rhs: number): Series<Bool8>;
  lt<R extends Numeric>(rhs: Scalar<R>): Series<Bool8>;
  lt<R extends Numeric>(rhs: Series<R>): Series<Bool8>;
  lt<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>) {
    switch (typeof rhs) {
      case 'bigint': return new Series(this._data.lt(rhs));
      case 'number': return new Series(this._data.lt(rhs));
      default: break;
    }
    return rhs instanceof Scalar ? new Series(this._data.lt(rhs))
                                 : new Series(this._data.lt(rhs._data));
  }

  /**
   * Perform the binary '<=' operation between this column and another Series or scalar value.
   *
   * @rhs The other Series or scalar to compare with this column.
   * @returns A Series of booleans with the comparison result.
   */
  le(rhs: bigint): Series<Bool8>;
  le(rhs: number): Series<Bool8>;
  le<R extends Numeric>(rhs: Scalar<R>): Series<Bool8>;
  le<R extends Numeric>(rhs: Series<R>): Series<Bool8>;
  le<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>) {
    switch (typeof rhs) {
      case 'bigint': return new Series(this._data.le(rhs));
      case 'number': return new Series(this._data.le(rhs));
      default: break;
    }
    return rhs instanceof Scalar ? new Series(this._data.le(rhs))
                                 : new Series(this._data.le(rhs._data));
  }

  /**
   * Perform the binary '>' operation between this column and another Series or scalar value.
   *
   * @rhs The other Series or scalar to compare with this column.
   * @returns A Series of booleans with the comparison result.
   */
  gt(rhs: bigint): Series<Bool8>;
  gt(rhs: number): Series<Bool8>;
  gt<R extends Numeric>(rhs: Scalar<R>): Series<Bool8>;
  gt<R extends Numeric>(rhs: Series<R>): Series<Bool8>;
  gt<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>) {
    switch (typeof rhs) {
      case 'bigint': return new Series(this._data.gt(rhs));
      case 'number': return new Series(this._data.gt(rhs));
      default: break;
    }
    return rhs instanceof Scalar ? new Series(this._data.gt(rhs))
                                 : new Series(this._data.gt(rhs._data));
  }

  /**
   * Perform the binary '>=' operation between this column and another Series or scalar value.
   *
   * @rhs The other Series or scalar to compare with this column.
   * @returns A Series of booleans with the comparison result.
   */
  ge(rhs: bigint): Series<Bool8>;
  ge(rhs: number): Series<Bool8>;
  ge<R extends Numeric>(rhs: Scalar<R>): Series<Bool8>;
  ge<R extends Numeric>(rhs: Series<R>): Series<Bool8>;
  ge<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>) {
    switch (typeof rhs) {
      case 'bigint': return new Series(this._data.ge(rhs));
      case 'number': return new Series(this._data.ge(rhs));
      default: break;
    }
    return rhs instanceof Scalar ? new Series(this._data.ge(rhs))
                                 : new Series(this._data.ge(rhs._data));
  }

  /**
   * Perform a binary `&` operation between this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to use.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  bitwise_and(rhs: bigint): Series<Int64>;
  bitwise_and(rhs: number): Series<Float64>;
  bitwise_and<R extends Numeric>(rhs: Scalar<R>): Series<CommonType<T, R>>;
  bitwise_and<R extends Numeric>(rhs: Series<R>): Series<CommonType<T, R>>;
  bitwise_and<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>) {
    switch (typeof rhs) {
      case 'bigint': return new Series(this._data.bitwise_and(rhs));
      case 'number': return new Series(this._data.bitwise_and(rhs));
      default: break;
    }
    return rhs instanceof Scalar ? new Series(this._data.bitwise_and(rhs))
                                 : new Series(this._data.bitwise_and(rhs._data));
  }

  /**
   * Perform a binary `|` operation between this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to use.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  bitwise_or(rhs: bigint): Series<Int64>;
  bitwise_or(rhs: number): Series<Float64>;
  bitwise_or<R extends Numeric>(rhs: Scalar<R>): Series<CommonType<T, R>>;
  bitwise_or<R extends Numeric>(rhs: Series<R>): Series<CommonType<T, R>>;
  bitwise_or<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>) {
    switch (typeof rhs) {
      case 'bigint': return new Series(this._data.bitwise_or(rhs));
      case 'number': return new Series(this._data.bitwise_or(rhs));
      default: break;
    }
    return rhs instanceof Scalar ? new Series(this._data.bitwise_or(rhs))
                                 : new Series(this._data.bitwise_or(rhs._data));
  }

  /**
   * Perform a binary `^` operation between this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to use.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  bitwise_xor(rhs: bigint): Series<Int64>;
  bitwise_xor(rhs: number): Series<Float64>;
  bitwise_xor<R extends Numeric>(rhs: Scalar<R>): Series<CommonType<T, R>>;
  bitwise_xor<R extends Numeric>(rhs: Series<R>): Series<CommonType<T, R>>;
  bitwise_xor<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>) {
    switch (typeof rhs) {
      case 'bigint': return new Series(this._data.bitwise_xor(rhs));
      case 'number': return new Series(this._data.bitwise_xor(rhs));
      default: break;
    }
    return rhs instanceof Scalar ? new Series(this._data.bitwise_xor(rhs))
                                 : new Series(this._data.bitwise_xor(rhs._data));
  }

  /**
   * Perform a binary `&&` operation between this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to use.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  logical_and(rhs: bigint): Series<Int64>;
  logical_and(rhs: number): Series<Float64>;
  logical_and<R extends Numeric>(rhs: Scalar<R>): Series<CommonType<T, R>>;
  logical_and<R extends Numeric>(rhs: Series<R>): Series<CommonType<T, R>>;
  logical_and<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>) {
    switch (typeof rhs) {
      case 'bigint': return new Series(this._data.logical_and(rhs));
      case 'number': return new Series(this._data.logical_and(rhs));
      default: break;
    }
    return rhs instanceof Scalar ? new Series(this._data.logical_and(rhs))
                                 : new Series(this._data.logical_and(rhs._data));
  }

  /**
   * Perform a binary `||` operation between this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to use.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  logical_or(rhs: bigint): Series<Int64>;
  logical_or(rhs: number): Series<Float64>;
  logical_or<R extends Numeric>(rhs: Scalar<R>): Series<CommonType<T, R>>;
  logical_or<R extends Numeric>(rhs: Series<R>): Series<CommonType<T, R>>;
  logical_or<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>) {
    switch (typeof rhs) {
      case 'bigint': return new Series(this._data.logical_or(rhs));
      case 'number': return new Series(this._data.logical_or(rhs));
      default: break;
    }
    return rhs instanceof Scalar ? new Series(this._data.logical_or(rhs))
                                 : new Series(this._data.logical_or(rhs._data));
  }

  /**
   * Perform a binary `coalesce` operation between this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to use.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  coalesce(rhs: bigint): Series<Int64>;
  coalesce(rhs: number): Series<Float64>;
  coalesce<R extends Numeric>(rhs: Scalar<R>): Series<CommonType<T, R>>;
  coalesce<R extends Numeric>(rhs: Series<R>): Series<CommonType<T, R>>;
  coalesce<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>) {
    switch (typeof rhs) {
      case 'bigint': return new Series(this._data.coalesce(rhs));
      case 'number': return new Series(this._data.coalesce(rhs));
      default: break;
    }
    return rhs instanceof Scalar ? new Series(this._data.coalesce(rhs))
                                 : new Series(this._data.coalesce(rhs._data));
  }

  /**
   * Perform a binary `<<` operation between this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to use.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  shift_left(rhs: bigint): Series<Int64>;
  shift_left(rhs: number): Series<Float64>;
  shift_left<R extends Numeric>(rhs: Scalar<R>): Series<CommonType<T, R>>;
  shift_left<R extends Numeric>(rhs: Series<R>): Series<CommonType<T, R>>;
  shift_left<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>) {
    switch (typeof rhs) {
      case 'bigint': return new Series(this._data.shift_left(rhs));
      case 'number': return new Series(this._data.shift_left(rhs));
      default: break;
    }
    return rhs instanceof Scalar ? new Series(this._data.shift_left(rhs))
                                 : new Series(this._data.shift_left(rhs._data));
  }

  /**
   * Perform a binary `>>` operation between this Series and another Series or scalar
   * value.
   *
   * @param rhs The other Series or scalar to use.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  shift_right(rhs: bigint): Series<Int64>;
  shift_right(rhs: number): Series<Float64>;
  shift_right<R extends Numeric>(rhs: Scalar<R>): Series<CommonType<T, R>>;
  shift_right<R extends Numeric>(rhs: Series<R>): Series<CommonType<T, R>>;
  shift_right<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>) {
    switch (typeof rhs) {
      case 'bigint': return new Series(this._data.shift_right(rhs));
      case 'number': return new Series(this._data.shift_right(rhs));
      default: break;
    }
    return rhs instanceof Scalar ? new Series(this._data.shift_right(rhs))
                                 : new Series(this._data.shift_right(rhs._data));
  }

  /**
   * Perform a binary `shift_right_unsigned` operation between this Series and another Series or
   * scalar value.
   *
   * @param rhs The other Series or scalar to use.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  shift_right_unsigned(rhs: bigint): Series<Int64>;
  shift_right_unsigned(rhs: number): Series<Float64>;
  shift_right_unsigned<R extends Numeric>(rhs: Scalar<R>): Series<CommonType<T, R>>;
  shift_right_unsigned<R extends Numeric>(rhs: Series<R>): Series<CommonType<T, R>>;
  shift_right_unsigned<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>) {
    switch (typeof rhs) {
      case 'bigint': return new Series(this._data.shift_right_unsigned(rhs));
      case 'number': return new Series(this._data.shift_right_unsigned(rhs));
      default: break;
    }
    return rhs instanceof Scalar ? new Series(this._data.shift_right_unsigned(rhs))
                                 : new Series(this._data.shift_right_unsigned(rhs._data));
  }

  /**
   * Perform a binary `log_base` operation between this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to use.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  log_base(rhs: bigint): Series<Int64>;
  log_base(rhs: number): Series<Float64>;
  log_base<R extends Numeric>(rhs: Scalar<R>): Series<CommonType<T, R>>;
  log_base<R extends Numeric>(rhs: Series<R>): Series<CommonType<T, R>>;
  log_base<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>) {
    switch (typeof rhs) {
      case 'bigint': return new Series(this._data.log_base(rhs));
      case 'number': return new Series(this._data.log_base(rhs));
      default: break;
    }
    return rhs instanceof Scalar ? new Series(this._data.log_base(rhs))
                                 : new Series(this._data.log_base(rhs._data));
  }

  /**
   * Perform a binary `atan2` operation between this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to use.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  atan2(rhs: bigint): Series<Int64>;
  atan2(rhs: number): Series<Float64>;
  atan2<R extends Numeric>(rhs: Scalar<R>): Series<CommonType<T, R>>;
  atan2<R extends Numeric>(rhs: Series<R>): Series<CommonType<T, R>>;
  atan2<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>) {
    switch (typeof rhs) {
      case 'bigint': return new Series(this._data.atan2(rhs));
      case 'number': return new Series(this._data.atan2(rhs));
      default: break;
    }
    return rhs instanceof Scalar ? new Series(this._data.atan2(rhs))
                                 : new Series(this._data.atan2(rhs._data));
  }

  /**
   * Perform a binary `null_equals` operation between this Series and another Series or scalar
   * value.
   *
   * @param rhs The other Series or scalar to use.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  null_equals(rhs: bigint): Series<Bool8>;
  null_equals(rhs: number): Series<Bool8>;
  null_equals<R extends Numeric>(rhs: Scalar<R>): Series<Bool8>;
  null_equals<R extends Numeric>(rhs: Series<R>): Series<Bool8>;
  null_equals<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>) {
    switch (typeof rhs) {
      case 'bigint': return new Series(this._data.null_equals(rhs));
      case 'number': return new Series(this._data.null_equals(rhs));
      default: break;
    }
    return rhs instanceof Scalar ? new Series(this._data.null_equals(rhs))
                                 : new Series(this._data.null_equals(rhs._data));
  }

  /**
   * Perform a binary `null_max` operation between this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to use.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  null_max(rhs: bigint): Series<Int64>;
  null_max(rhs: number): Series<Float64>;
  null_max<R extends Numeric>(rhs: Scalar<R>): Series<CommonType<T, R>>;
  null_max<R extends Numeric>(rhs: Series<R>): Series<CommonType<T, R>>;
  null_max<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>) {
    switch (typeof rhs) {
      case 'bigint': return new Series(this._data.null_max(rhs));
      case 'number': return new Series(this._data.null_max(rhs));
      default: break;
    }
    return rhs instanceof Scalar ? new Series(this._data.null_max(rhs))
                                 : new Series(this._data.null_max(rhs._data));
  }

  /**
   * Perform a binary `null_min` operation between this Series and another Series or scalar value.
   *
   * @param rhs The other Series or scalar to use.
   * @returns A Series of a common numeric type with the results of the binary operation.
   */
  null_min(rhs: bigint): Series<Int64>;
  null_min(rhs: number): Series<Float64>;
  null_min<R extends Numeric>(rhs: Scalar<R>): Series<CommonType<T, R>>;
  null_min<R extends Numeric>(rhs: Series<R>): Series<CommonType<T, R>>;
  null_min<R extends Numeric>(rhs: bigint|number|Scalar<R>|Series<R>) {
    switch (typeof rhs) {
      case 'bigint': return new Series(this._data.null_min(rhs));
      case 'number': return new Series(this._data.null_min(rhs));
      default: break;
    }
    return rhs instanceof Scalar ? new Series(this._data.null_min(rhs))
                                 : new Series(this._data.null_min(rhs._data));
  }
}
