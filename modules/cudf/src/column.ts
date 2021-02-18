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
import {DeviceBuffer, MemoryResource} from '@nvidia/rmm';

import CUDF from './addon';
import {Scalar} from './scalar';
import {
  Bool8,
  DataType,
  Float64,
  Int64,
  Integral,
  Numeric,
  Uint64,
} from './types/dtypes';
import {CommonType, Interpolation} from './types/mappings';

export type ColumnProps<T extends DataType = any> = {
  // todo -- need to pass full DataType instance when we implement fixed_point
  type: T,
  data?: DeviceBuffer|MemoryData|number[]|null,
  offset?: number,
  length?: number,
  nullCount?: number,
  nullMask?: DeviceBuffer|MemoryData|number[]|boolean|null,
  children?: ReadonlyArray<Column>|null
};

interface ColumnConstructor {
  readonly prototype: Column;
  new<T extends DataType = any>(props: ColumnProps<T>): Column<T>;
}

/**
 * A low-level wrapper for libcudf Column Objects
 */
export interface Column<T extends DataType = any> {
  readonly type: T;
  readonly data: DeviceBuffer;
  readonly mask: DeviceBuffer;

  readonly length: number;
  readonly nullable: boolean;
  readonly hasNulls: boolean;
  readonly nullCount: number;
  readonly numChildren: number;

  /**
   * Return sub-selection from a Column
   *
   * @param selection
   */
  gather(selection: Column<Integral|Bool8>): Column<T>;

  /**
   * Return a child at the specified index to host memory
   *
   * @param index
   */
  getChild<R extends DataType>(index: number): Column<R>;

  /**
   * Return a value at the specified index to host memory
   *
   * @param index
   */
  getValue(index: number): T['scalarType']|null;

  // setValue(index: number, value?: T['scalarType'] | null): void;

  /**
   * Set the null count for the null mask
   *
   * @param nullCount
   */
  setNullCount(nullCount: number): void;

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
  setNullMask(mask: DeviceBuffer, nullCount?: number): void;

  /**
   * Add this Column and another Column or scalar value.
   *
   * @param rhs The other Column or scalar to add to this Column.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  add(rhs: bigint, memoryResource?: MemoryResource): Column<Int64>;
  add(rhs: number, memoryResource?: MemoryResource): Column<Float64>;
  add<R extends Numeric>(rhs: Scalar<R>, memoryResource?: MemoryResource): Column<CommonType<T, R>>;
  add<R extends Numeric>(rhs: Column<R>, memoryResource?: MemoryResource): Column<CommonType<T, R>>;

  /**
   * Subtract this Column and another Column or scalar value.
   *
   * @param rhs The other Column or scalar to subtract from this Column.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  sub(rhs: bigint, memoryResource?: MemoryResource): Column<Int64>;
  sub(rhs: number, memoryResource?: MemoryResource): Column<Float64>;
  sub<R extends Numeric>(rhs: Scalar<R>, memoryResource?: MemoryResource): Column<CommonType<T, R>>;
  sub<R extends Numeric>(rhs: Column<R>, memoryResource?: MemoryResource): Column<CommonType<T, R>>;

  /**
   * Multiply this Column and another Column or scalar value.
   *
   * @param rhs The other Column or scalar to multiply this column by.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  mul(rhs: bigint, memoryResource?: MemoryResource): Column<Int64>;
  mul(rhs: number, memoryResource?: MemoryResource): Column<Float64>;
  mul<R extends Numeric>(rhs: Scalar<R>, memoryResource?: MemoryResource): Column<CommonType<T, R>>;
  mul<R extends Numeric>(rhs: Column<R>, memoryResource?: MemoryResource): Column<CommonType<T, R>>;

  /**
   * Divide this Column and another Column or scalar value.
   *
   * @param rhs The other Column or scalar to divide this Column by.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  div(rhs: bigint, memoryResource?: MemoryResource): Column<Int64>;
  div(rhs: number, memoryResource?: MemoryResource): Column<Float64>;
  div<R extends Numeric>(rhs: Scalar<R>, memoryResource?: MemoryResource): Column<CommonType<T, R>>;
  div<R extends Numeric>(rhs: Column<R>, memoryResource?: MemoryResource): Column<CommonType<T, R>>;

  /**
   * True-divide this Column and another Column or scalar value.
   *
   * @param rhs The other Column or scalar to true-divide this Column by.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  true_div(rhs: bigint, memoryResource?: MemoryResource): Column<Int64>;
  true_div(rhs: number, memoryResource?: MemoryResource): Column<Float64>;
  true_div<R extends Numeric>(rhs: Scalar<R>,
                              memoryResource?: MemoryResource): Column<CommonType<T, R>>;
  true_div<R extends Numeric>(rhs: Column<R>,
                              memoryResource?: MemoryResource): Column<CommonType<T, R>>;

  /**
   * Floor-divide this Column and another Column or scalar value.
   *
   * @param rhs The other Column or scalar to floor-divide this Column by.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  floor_div(rhs: bigint, memoryResource?: MemoryResource): Column<Int64>;
  floor_div(rhs: number, memoryResource?: MemoryResource): Column<Float64>;
  floor_div<R extends Numeric>(rhs: Scalar<R>,
                               memoryResource?: MemoryResource): Column<CommonType<T, R>>;
  floor_div<R extends Numeric>(rhs: Column<R>,
                               memoryResource?: MemoryResource): Column<CommonType<T, R>>;

  /**
   * Modulo this Column and another Column or scalar value.
   *
   * @param rhs The other Column or scalar to mod with this Column.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  mod(rhs: bigint, memoryResource?: MemoryResource): Column<Int64>;
  mod(rhs: number, memoryResource?: MemoryResource): Column<Float64>;
  mod<R extends Numeric>(rhs: Scalar<R>, memoryResource?: MemoryResource): Column<CommonType<T, R>>;
  mod<R extends Numeric>(rhs: Column<R>, memoryResource?: MemoryResource): Column<CommonType<T, R>>;

  /**
   * Power this Column and another Column or scalar value.
   *
   * @param rhs The other Column or scalar to use as the exponent for the power operation.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  pow(rhs: bigint, memoryResource?: MemoryResource): Column<Int64>;
  pow(rhs: number, memoryResource?: MemoryResource): Column<Float64>;
  pow<R extends Numeric>(rhs: Scalar<R>, memoryResource?: MemoryResource): Column<CommonType<T, R>>;
  pow<R extends Numeric>(rhs: Column<R>, memoryResource?: MemoryResource): Column<CommonType<T, R>>;

  /**
   * Perform the binary '==' operation between this column and another Column or scalar value.
   *
   * @rhs The other Column or scalar to compare with this column.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of booleans with the comparison result.
   */
  eq(rhs: bigint, memoryResource?: MemoryResource): Column<Bool8>;
  eq(rhs: number, memoryResource?: MemoryResource): Column<Bool8>;
  eq<R extends Numeric>(rhs: Scalar<R>, memoryResource?: MemoryResource): Column<Bool8>;
  eq<R extends Numeric>(rhs: Column<R>, memoryResource?: MemoryResource): Column<Bool8>;

  /**
   * Perform the binary '!=' operation between this column and another Column or scalar value.
   *
   * @rhs The other Column or scalar to compare with this column.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of booleans with the comparison result.
   */
  ne(rhs: bigint, memoryResource?: MemoryResource): Column<Bool8>;
  ne(rhs: number, memoryResource?: MemoryResource): Column<Bool8>;
  ne<R extends Numeric>(rhs: Scalar<R>, memoryResource?: MemoryResource): Column<Bool8>;
  ne<R extends Numeric>(rhs: Column<R>, memoryResource?: MemoryResource): Column<Bool8>;

  /**
   * Perform the binary '<' operation between this column and another Column or scalar value.
   *
   * @rhs The other Column or scalar to compare with this column.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of booleans with the comparison result.
   */
  lt(rhs: bigint, memoryResource?: MemoryResource): Column<Bool8>;
  lt(rhs: number, memoryResource?: MemoryResource): Column<Bool8>;
  lt<R extends Numeric>(rhs: Scalar<R>, memoryResource?: MemoryResource): Column<Bool8>;
  lt<R extends Numeric>(rhs: Column<R>, memoryResource?: MemoryResource): Column<Bool8>;

  /**
   * Perform the binary '<=' operation between this column and another Column or scalar value.
   *
   * @rhs The other Column or scalar to compare with this column.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of booleans with the comparison result.
   */
  le(rhs: bigint, memoryResource?: MemoryResource): Column<Bool8>;
  le(rhs: number, memoryResource?: MemoryResource): Column<Bool8>;
  le<R extends Numeric>(rhs: Scalar<R>, memoryResource?: MemoryResource): Column<Bool8>;
  le<R extends Numeric>(rhs: Column<R>, memoryResource?: MemoryResource): Column<Bool8>;

  /**
   * Perform the binary '>' operation between this column and another Column or scalar value.
   *
   * @rhs The other Column or scalar to compare with this column.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of booleans with the comparison result.
   */
  gt(rhs: bigint, memoryResource?: MemoryResource): Column<Bool8>;
  gt(rhs: number, memoryResource?: MemoryResource): Column<Bool8>;
  gt<R extends Numeric>(rhs: Scalar<R>, memoryResource?: MemoryResource): Column<Bool8>;
  gt<R extends Numeric>(rhs: Column<R>, memoryResource?: MemoryResource): Column<Bool8>;

  /**
   * Perform the binary '>=' operation between this column and another Column or scalar value.
   *
   * @rhs The other Column or scalar to compare with this column.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of booleans with the comparison result.
   */
  ge(rhs: bigint, memoryResource?: MemoryResource): Column<Bool8>;
  ge(rhs: number, memoryResource?: MemoryResource): Column<Bool8>;
  ge<R extends Numeric>(rhs: Scalar<R>, memoryResource?: MemoryResource): Column<Bool8>;
  ge<R extends Numeric>(rhs: Column<R>, memoryResource?: MemoryResource): Column<Bool8>;

  /**
   * Perform a binary `&` operation between this Column and another Column or scalar value.
   *
   * @param rhs The other Column or scalar to use.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  bitwise_and(rhs: bigint, memoryResource?: MemoryResource): Column<T>;
  bitwise_and(rhs: number, memoryResource?: MemoryResource): Column<T>;
  bitwise_and<R extends Numeric>(rhs: Scalar<R>,
                                 memoryResource?: MemoryResource): Column<CommonType<T, R>>;
  bitwise_and<R extends Numeric>(rhs: Column<R>,
                                 memoryResource?: MemoryResource): Column<CommonType<T, R>>;

  /**
   * Perform a binary `|` operation between this Column and another Column or scalar value.
   *
   * @param rhs The other Column or scalar to use.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  bitwise_or(rhs: bigint, memoryResource?: MemoryResource): Column<T>;
  bitwise_or(rhs: number, memoryResource?: MemoryResource): Column<T>;
  bitwise_or<R extends Numeric>(rhs: Scalar<R>,
                                memoryResource?: MemoryResource): Column<CommonType<T, R>>;
  bitwise_or<R extends Numeric>(rhs: Column<R>,
                                memoryResource?: MemoryResource): Column<CommonType<T, R>>;

  /**
   * Perform a binary `^` operation between this Column and another Column or scalar value.
   *
   * @param rhs The other Column or scalar to use.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  bitwise_xor(rhs: bigint, memoryResource?: MemoryResource): Column<T>;
  bitwise_xor(rhs: number, memoryResource?: MemoryResource): Column<T>;
  bitwise_xor<R extends Numeric>(rhs: Scalar<R>,
                                 memoryResource?: MemoryResource): Column<CommonType<T, R>>;
  bitwise_xor<R extends Numeric>(rhs: Column<R>,
                                 memoryResource?: MemoryResource): Column<CommonType<T, R>>;

  /**
   * Perform a binary `&&` operation between this Column and another Column or scalar value.
   *
   * @param rhs The other Column or scalar to use.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  logical_and(rhs: bigint, memoryResource?: MemoryResource): Column<Int64>;
  logical_and(rhs: number, memoryResource?: MemoryResource): Column<Float64>;
  logical_and<R extends Numeric>(rhs: Scalar<R>,
                                 memoryResource?: MemoryResource): Column<CommonType<T, R>>;
  logical_and<R extends Numeric>(rhs: Column<R>,
                                 memoryResource?: MemoryResource): Column<CommonType<T, R>>;

  /**
   * Perform a binary `||` operation between this Column and another Column or scalar value.
   *
   * @param rhs The other Column or scalar to use.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  logical_or(rhs: bigint, memoryResource?: MemoryResource): Column<Int64>;
  logical_or(rhs: number, memoryResource?: MemoryResource): Column<Float64>;
  logical_or<R extends Numeric>(rhs: Scalar<R>,
                                memoryResource?: MemoryResource): Column<CommonType<T, R>>;
  logical_or<R extends Numeric>(rhs: Column<R>,
                                memoryResource?: MemoryResource): Column<CommonType<T, R>>;

  /**
   * Perform a binary `coalesce` operation between this Column and another Column or scalar value.
   *
   * @param rhs The other Column or scalar to use.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  coalesce(rhs: bigint, memoryResource?: MemoryResource): Column<Int64>;
  coalesce(rhs: number, memoryResource?: MemoryResource): Column<Float64>;
  coalesce<R extends Numeric>(rhs: Scalar<R>,
                              memoryResource?: MemoryResource): Column<CommonType<T, R>>;
  coalesce<R extends Numeric>(rhs: Column<R>,
                              memoryResource?: MemoryResource): Column<CommonType<T, R>>;

  /**
   * Perform a binary `<<` operation between this Column and another Column or scalar value.
   *
   * @param rhs The other Column or scalar to use.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  shift_left(rhs: bigint, memoryResource?: MemoryResource): Column<T>;
  shift_left(rhs: number, memoryResource?: MemoryResource): Column<T>;
  shift_left<R extends Numeric>(rhs: Scalar<R>,
                                memoryResource?: MemoryResource): Column<CommonType<T, R>>;
  shift_left<R extends Numeric>(rhs: Column<R>,
                                memoryResource?: MemoryResource): Column<CommonType<T, R>>;

  /**
   * Perform a binary `>>` operation between this Column and another Column or scalar
   * value.
   *
   * @param rhs The other Column or scalar to use.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  shift_right(rhs: bigint, memoryResource?: MemoryResource): Column<T>;
  shift_right(rhs: number, memoryResource?: MemoryResource): Column<T>;
  shift_right<R extends Numeric>(rhs: Scalar<R>,
                                 memoryResource?: MemoryResource): Column<CommonType<T, R>>;
  shift_right<R extends Numeric>(rhs: Column<R>,
                                 memoryResource?: MemoryResource): Column<CommonType<T, R>>;

  /**
   * Perform a binary `shift_right_unsigned` operation between this Column and another Column or
   * scalar value.
   *
   * @param rhs The other Column or scalar to use.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  shift_right_unsigned(rhs: bigint, memoryResource?: MemoryResource): Column<T>;
  shift_right_unsigned(rhs: number, memoryResource?: MemoryResource): Column<T>;
  shift_right_unsigned<R extends Numeric>(rhs: Scalar<R>, memoryResource?: MemoryResource):
    Column<CommonType<T, R>>;
  shift_right_unsigned<R extends Numeric>(rhs: Column<R>, memoryResource?: MemoryResource):
    Column<CommonType<T, R>>;

  /**
   * Perform a binary `log_base` operation between this Column and another Column or scalar value.
   *
   * @param rhs The other Column or scalar to use.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  log_base(rhs: bigint, memoryResource?: MemoryResource): Column<Int64>;
  log_base(rhs: number, memoryResource?: MemoryResource): Column<Float64>;
  log_base<R extends Numeric>(rhs: Scalar<R>,
                              memoryResource?: MemoryResource): Column<CommonType<T, R>>;
  log_base<R extends Numeric>(rhs: Column<R>,
                              memoryResource?: MemoryResource): Column<CommonType<T, R>>;

  /**
   * Perform a binary `atan2` operation between this Column and another Column or scalar value.
   *
   * @param rhs The other Column or scalar to use.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  atan2(rhs: bigint, memoryResource?: MemoryResource): Column<Int64>;
  atan2(rhs: number, memoryResource?: MemoryResource): Column<Float64>;
  atan2<R extends Numeric>(rhs: Scalar<R>,
                           memoryResource?: MemoryResource): Column<CommonType<T, R>>;
  atan2<R extends Numeric>(rhs: Column<R>,
                           memoryResource?: MemoryResource): Column<CommonType<T, R>>;

  /**
   * Perform a binary `null_equals` operation between this Column and another Column or scalar
   * value.
   *
   * @param rhs The other Column or scalar to use.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  null_equals(rhs: bigint, memoryResource?: MemoryResource): Column<Bool8>;
  null_equals(rhs: number, memoryResource?: MemoryResource): Column<Bool8>;
  null_equals<R extends Numeric>(rhs: Scalar<R>, memoryResource?: MemoryResource): Column<Bool8>;
  null_equals<R extends Numeric>(rhs: Column<R>, memoryResource?: MemoryResource): Column<Bool8>;

  /**
   * Perform a binary `null_max` operation between this Column and another Column or scalar value.
   *
   * @param rhs The other Column or scalar to use.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  null_max(rhs: bigint, memoryResource?: MemoryResource): Column<Int64>;
  null_max(rhs: number, memoryResource?: MemoryResource): Column<Float64>;
  null_max<R extends Numeric>(rhs: Scalar<R>,
                              memoryResource?: MemoryResource): Column<CommonType<T, R>>;
  null_max<R extends Numeric>(rhs: Column<R>,
                              memoryResource?: MemoryResource): Column<CommonType<T, R>>;

  /**
   * Perform a binary `null_min` operation between this Column and another Column or scalar value.
   *
   * @param rhs The other Column or scalar to use.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  null_min(rhs: bigint, memoryResource?: MemoryResource): Column<Int64>;
  null_min(rhs: number, memoryResource?: MemoryResource): Column<Float64>;
  null_min<R extends Numeric>(rhs: Scalar<R>,
                              memoryResource?: MemoryResource): Column<CommonType<T, R>>;
  null_min<R extends Numeric>(rhs: Column<R>,
                              memoryResource?: MemoryResource): Column<CommonType<T, R>>;

  /**
   * Casts data from dtype specified in input to dtype specified in output.
   *
   * @note Supports only fixed-width types.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns Column of same size as `input` containing result of the cast operation.
   */
  cast<R extends DataType>(dataType: R, memoryResource?: MemoryResource): Column<R>;

  /**
   * Creates a column of `BOOL8` elements where `true` indicates the value is null and `false`
   * indicates the value is valid.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A non-nullable column of `BOOL8` elements with `true` representing `null`
   *   values.
   */
  isNull(memoryResource?: MemoryResource): Column<Bool8>;

  /**
   * Creates a column of `BOOL8` elements where `true` indicates the value is valid and `false`
   * indicates the value is null.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A non-nullable column of `BOOL8` elements with `false` representing `null`
   *   values.
   */
  isValid(memoryResource?: MemoryResource): Column<Bool8>;

  /**
   * Creates a column of `BOOL8` elements indicating the presence of `NaN` values in a
   * column of floating point values. The output element at row `i` is `true` if the element in
   * `input` at row i is `NAN`, else `false`
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A non-nullable column of `BOOL8` elements with `true` representing `NAN`
   *   values
   */
  isNaN(memoryResource?: MemoryResource): Column<Bool8>;

  /**
   * Creates a column of `BOOL8` elements indicating the absence of `NaN` values in a
   * column of floating point values. The output element at row `i` is `false` if the element in
   * `input` at row i is `NAN`, else `true`
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A non-nullable column of `BOOL8` elements with `true` representing non-`NAN`
   *   values
   */
  isNotNaN(memoryResource?: MemoryResource): Column<Bool8>;

  /**
   * Compute the trigonometric sine for each value in this Column.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of the same number of elements containing the result of the operation.
   */
  sin(memoryResource?: MemoryResource): Column<T>;

  /**
   * Compute the trigonometric cosine for each value in this Column.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of the same number of elements containing the result of the operation.
   */
  cos(memoryResource?: MemoryResource): Column<T>;

  /**
   * Compute the trigonometric tangent for each value in this Column.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of the same number of elements containing the result of the operation.
   */
  tan(memoryResource?: MemoryResource): Column<T>;

  /**
   * Compute the trigonometric sine inverse for each value in this Column.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of the same number of elements containing the result of the operation.
   */
  asin(memoryResource?: MemoryResource): Column<T>;

  /**
   * Compute the trigonometric cosine inverse for each value in this Column.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of the same number of elements containing the result of the operation.
   */
  acos(memoryResource?: MemoryResource): Column<T>;

  /**
   * Compute the trigonometric tangent inverse for each value in this Column.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of the same number of elements containing the result of the operation.
   */
  atan(memoryResource?: MemoryResource): Column<T>;

  /**
   * Compute the hyperbolic sine for each value in this Column.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of the same number of elements containing the result of the operation.
   */
  sinh(memoryResource?: MemoryResource): Column<T>;

  /**
   * Compute the hyperbolic cosine for each value in this Column.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of the same number of elements containing the result of the operation.
   */
  cosh(memoryResource?: MemoryResource): Column<T>;

  /**
   * Compute the hyperbolic tangent for each value in this Column.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of the same number of elements containing the result of the operation.
   */
  tanh(memoryResource?: MemoryResource): Column<T>;

  /**
   * Compute the hyperbolic sine inverse for each value in this Column.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of the same number of elements containing the result of the operation.
   */
  asinh(memoryResource?: MemoryResource): Column<T>;

  /**
   * Compute the hyperbolic cosine inverse for each value in this Column.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of the same number of elements containing the result of the operation.
   */
  acosh(memoryResource?: MemoryResource): Column<T>;

  /**
   * Compute the hyperbolic tangent inverse for each value in this Column.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of the same number of elements containing the result of the operation.
   */
  atanh(memoryResource?: MemoryResource): Column<T>;

  /**
   * Compute the exponential (base e, euler number) for each value in this Column.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of the same number of elements containing the result of the operation.
   */
  exp(memoryResource?: MemoryResource): Column<T>;

  /**
   * Compute the natural logarithm (base e) for each value in this Column.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of the same number of elements containing the result of the operation.
   */
  log(memoryResource?: MemoryResource): Column<T>;

  /**
   * Compute the square-root (x^0.5) for each value in this Column.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of the same number of elements containing the result of the operation.
   */
  sqrt(memoryResource?: MemoryResource): Column<T>;

  /**
   * Compute the cube-root (x^(1.0/3)) for each value in this Column.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of the same number of elements containing the result of the operation.
   */
  cbrt(memoryResource?: MemoryResource): Column<T>;

  /**
   * Compute the smallest integer value not less than arg for each value in this Column.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of the same number of elements containing the result of the operation.
   */
  ceil(memoryResource?: MemoryResource): Column<T>;

  /**
   * Compute the largest integer value not greater than arg for each value in this Column.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of the same number of elements containing the result of the operation.
   */
  floor(memoryResource?: MemoryResource): Column<T>;

  /**
   * Compute the absolute value for each value in this Column.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of the same number of elements containing the result of the operation.
   */
  abs(memoryResource?: MemoryResource): Column<T>;

  /**
   * Round floating-point to integer for each value in this Column.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @throws cudf::logic_error if the Column's DataType isn't Float32 or Float64.
   * @returns A Column of the same number of elements containing the result of the operation.
   */
  rint(memoryResource?: MemoryResource): Column<T>;

  /**
   * Compute the bitwise not (~) for each value in this Column.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of the same number of elements containing the result of the operation.
   */
  bit_invert(memoryResource?: MemoryResource): Column<T>;

  /**
   * Compute the logical not (!) for each value in this Column.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns A Column of the same number of elements containing the result of the operation.
   */
  not(memoryResource?: MemoryResource): Column<Bool8>;

  /**
   * Compute the sum of all values in this Column.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns The sum of all the values in this Column.
   */
  sum(memoryResource?: MemoryResource): T extends(Integral|Int64|Uint64)? bigint: number;

  /**
   * Compute the product of all values in this Column.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns The product of all the values in this Column.
   */
  product(memoryResource?: MemoryResource): T extends(Integral|Int64|Uint64)? bigint: number;

  /**
   * Compute the sum_of_squares of all values in this Column.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns The sum_of_squares of all the values in this Column.
   */
  sum_of_squares(memoryResource?: MemoryResource): T extends(Integral|Int64|Uint64)? bigint: number;

  /**
   * Compute the mean of all values in this Column.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns The mean of all the values in this Column.
   */
  mean(memoryResource?: MemoryResource): number;

  /**
   * Compute the median of all values in this Column.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns The median of all the values in this Column.
   */
  median(memoryResource?: MemoryResource): number;

  /**
   * Compute the nunique of all values in this Column.
   *
   * @param skipna The skipna parameter if true, includes nulls while computing nunique,
   * if false, excludes the nulls while computing nunique.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns The number of unique values in this Column.
   */
  nunique(skipna?: boolean, memoryResource?: MemoryResource): number;

  /**
   * Return whether all elements are true in column.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns true if all elements are true in column, else false.
   */
  all(memoryResource?: MemoryResource): boolean;

  /**
   * Return whether any elements are true in column.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns true if any elements are true in column, else false.
   */
  any(memoryResource?: MemoryResource): boolean;

  /**
   * Return unbiased variance of the column.
   * Normalized by N-1 by default. This can be changed using the `ddof` argument
   *
   * @param ddof Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
   *  where N represents the number of elements.
   * @param memoryResource The optional MemoryResource used to allocate the result column's device
   *   memory.
   * @returns The median of all the values in this column.
   */
  var(ddof?: number, memoryResource?: MemoryResource): number;

  /**
   * Return sample standard deviation of the column.
   * Normalized by N-1 by default. This can be changed using the `ddof` argument
   *
   * @param ddof Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
   *  where N represents the number of elements.
   * @param memoryResource The optional MemoryResource used to allocate the result column's device
   *   memory.
   * @returns The median of all the values in this column.
   */
  std(ddof?: number, memoryResource?: MemoryResource): number;

  /**
   * Return values at the given quantile.
   *
   * @param q the quantile(s) to compute, 0 <= q <= 1
   * @param interpolation This optional parameter specifies the interpolation method to use,
   *  when the desired quantile lies between two data points i and j.
   * @param memoryResource The optional MemoryResource used to allocate the result column's device
   *   memory.
   * @returns values at the given quantile.
   */
  quantile(q?: number, interpolation?: Interpolation, memoryResource?: MemoryResource): number;

  /**
   * drop NA values from the column if column is of floating-type
   * values and contains NA values
   * @param memoryResource The optional MemoryResource used to allocate the result column's device
   *   memory.
   * @returns column without NaN and Null values
   */
  drop_nulls(memoryResource?: MemoryResource): Column<T>;

  /**
   * drop NA values from the column if column is of floating-type
   * values and contains NA values
   * @param memoryResource The optional MemoryResource used to allocate the result column's device
   *   memory.
   * @returns column without NaN and Null values
   */
  drop_nans(memoryResource?: MemoryResource): Column<T>;

  /**
   * convert NaN values in the column with Null values,
   * while also updating the nullMask and nullCount values
   *
   * @param memoryResource The optional MemoryResource used to allocate the result column's device
   *   memory.
   * @returns undefined if inplace=True, else updated column with Null values
   */
  nans_to_nulls(memoryResource?: MemoryResource): Column<T>;
}

// eslint-disable-next-line @typescript-eslint/no-redeclare
export const Column: ColumnConstructor = CUDF.Column;
