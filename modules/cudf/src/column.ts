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

import CUDF from './addon';
import {Scalar} from './scalar';
import {Bool8, CommonType, DataType, Float64, Int64, Integral, Numeric} from './types';

export type ColumnProps<T extends DataType = any> = {
  // todo -- need to pass full DataType instance when we implement fixed_point
  type: T['id'],
  data?: DeviceBuffer|MemoryData|null,
  offset?: number,
  length?: number,
  nullCount?: number,
  nullMask?: DeviceBuffer|MemoryData|null,
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
  getChild(index: number): Column;

  /**
   * Return a value at the specified index to host memory
   *
   * @param index
   */
  getValue(index: number): T['valueType']|null;

  // setValue(index: number, value?: T['valueType'] | null): void;

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
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  add(rhs: bigint): Column<Int64>;
  add(rhs: number): Column<Float64>;
  add<R extends Numeric>(rhs: Scalar<R>): Column<CommonType<T, R>>;
  add<R extends Numeric>(rhs: Column<R>): Column<CommonType<T, R>>;

  /**
   * Subtract this Column and another Column or scalar value.
   *
   * @param rhs The other Column or scalar to subtract from this Column.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  sub(rhs: bigint): Column<Int64>;
  sub(rhs: number): Column<Float64>;
  sub<R extends Numeric>(rhs: Scalar<R>): Column<CommonType<T, R>>;
  sub<R extends Numeric>(rhs: Column<R>): Column<CommonType<T, R>>;

  /**
   * Multiply this Column and another Column or scalar value.
   *
   * @param rhs The other Column or scalar to multiply this column by.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  mul(rhs: bigint): Column<Int64>;
  mul(rhs: number): Column<Float64>;
  mul<R extends Numeric>(rhs: Scalar<R>): Column<CommonType<T, R>>;
  mul<R extends Numeric>(rhs: Column<R>): Column<CommonType<T, R>>;

  /**
   * Divide this Column and another Column or scalar value.
   *
   * @param rhs The other Column or scalar to divide this Column by.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  div(rhs: bigint): Column<Int64>;
  div(rhs: number): Column<Float64>;
  div<R extends Numeric>(rhs: Scalar<R>): Column<CommonType<T, R>>;
  div<R extends Numeric>(rhs: Column<R>): Column<CommonType<T, R>>;

  /**
   * True-divide this Column and another Column or scalar value.
   *
   * @param rhs The other Column or scalar to true-divide this Column by.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  true_div(rhs: bigint): Column<Int64>;
  true_div(rhs: number): Column<Float64>;
  true_div<R extends Numeric>(rhs: Scalar<R>): Column<CommonType<T, R>>;
  true_div<R extends Numeric>(rhs: Column<R>): Column<CommonType<T, R>>;

  /**
   * Floor-divide this Column and another Column or scalar value.
   *
   * @param rhs The other Column or scalar to floor-divide this Column by.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  floor_div(rhs: bigint): Column<Int64>;
  floor_div(rhs: number): Column<Float64>;
  floor_div<R extends Numeric>(rhs: Scalar<R>): Column<CommonType<T, R>>;
  floor_div<R extends Numeric>(rhs: Column<R>): Column<CommonType<T, R>>;

  /**
   * Modulo this Column and another Column or scalar value.
   *
   * @param rhs The other Column or scalar to mod with this Column.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  mod(rhs: bigint): Column<Int64>;
  mod(rhs: number): Column<Float64>;
  mod<R extends Numeric>(rhs: Scalar<R>): Column<CommonType<T, R>>;
  mod<R extends Numeric>(rhs: Column<R>): Column<CommonType<T, R>>;

  /**
   * Power this Column and another Column or scalar value.
   *
   * @param rhs The other Column or scalar to use as the exponent for the power operation.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  pow(rhs: bigint): Column<Int64>;
  pow(rhs: number): Column<Float64>;
  pow<R extends Numeric>(rhs: Scalar<R>): Column<CommonType<T, R>>;
  pow<R extends Numeric>(rhs: Column<R>): Column<CommonType<T, R>>;

  /**
   * Perform the binary '==' operation between this column and another Column or scalar value.
   *
   * @rhs The other Column or scalar to compare with this column.
   * @returns A Column of booleans with the comparison result.
   */
  eq(rhs: bigint): Column<Bool8>;
  eq(rhs: number): Column<Bool8>;
  eq<R extends Numeric>(rhs: Scalar<R>): Column<Bool8>;
  eq<R extends Numeric>(rhs: Column<R>): Column<Bool8>;

  /**
   * Perform the binary '!=' operation between this column and another Column or scalar value.
   *
   * @rhs The other Column or scalar to compare with this column.
   * @returns A Column of booleans with the comparison result.
   */
  ne(rhs: bigint): Column<Bool8>;
  ne(rhs: number): Column<Bool8>;
  ne<R extends Numeric>(rhs: Scalar<R>): Column<Bool8>;
  ne<R extends Numeric>(rhs: Column<R>): Column<Bool8>;

  /**
   * Perform the binary '<' operation between this column and another Column or scalar value.
   *
   * @rhs The other Column or scalar to compare with this column.
   * @returns A Column of booleans with the comparison result.
   */
  lt(rhs: bigint): Column<Bool8>;
  lt(rhs: number): Column<Bool8>;
  lt<R extends Numeric>(rhs: Scalar<R>): Column<Bool8>;
  lt<R extends Numeric>(rhs: Column<R>): Column<Bool8>;

  /**
   * Perform the binary '<=' operation between this column and another Column or scalar value.
   *
   * @rhs The other Column or scalar to compare with this column.
   * @returns A Column of booleans with the comparison result.
   */
  le(rhs: bigint): Column<Bool8>;
  le(rhs: number): Column<Bool8>;
  le<R extends Numeric>(rhs: Scalar<R>): Column<Bool8>;
  le<R extends Numeric>(rhs: Column<R>): Column<Bool8>;

  /**
   * Perform the binary '>' operation between this column and another Column or scalar value.
   *
   * @rhs The other Column or scalar to compare with this column.
   * @returns A Column of booleans with the comparison result.
   */
  gt(rhs: bigint): Column<Bool8>;
  gt(rhs: number): Column<Bool8>;
  gt<R extends Numeric>(rhs: Scalar<R>): Column<Bool8>;
  gt<R extends Numeric>(rhs: Column<R>): Column<Bool8>;

  /**
   * Perform the binary '>=' operation between this column and another Column or scalar value.
   *
   * @rhs The other Column or scalar to compare with this column.
   * @returns A Column of booleans with the comparison result.
   */
  ge(rhs: bigint): Column<Bool8>;
  ge(rhs: number): Column<Bool8>;
  ge<R extends Numeric>(rhs: Scalar<R>): Column<Bool8>;
  ge<R extends Numeric>(rhs: Column<R>): Column<Bool8>;

  /**
   * Perform a binary `&` operation between this Column and another Column or scalar value.
   *
   * @param rhs The other Column or scalar to use.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  bitwise_and(rhs: bigint): Column<Int64>;
  bitwise_and(rhs: number): Column<Float64>;
  bitwise_and<R extends Numeric>(rhs: Scalar<R>): Column<CommonType<T, R>>;
  bitwise_and<R extends Numeric>(rhs: Column<R>): Column<CommonType<T, R>>;

  /**
   * Perform a binary `|` operation between this Column and another Column or scalar value.
   *
   * @param rhs The other Column or scalar to use.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  bitwise_or(rhs: bigint): Column<Int64>;
  bitwise_or(rhs: number): Column<Float64>;
  bitwise_or<R extends Numeric>(rhs: Scalar<R>): Column<CommonType<T, R>>;
  bitwise_or<R extends Numeric>(rhs: Column<R>): Column<CommonType<T, R>>;

  /**
   * Perform a binary `^` operation between this Column and another Column or scalar value.
   *
   * @param rhs The other Column or scalar to use.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  bitwise_xor(rhs: bigint): Column<Int64>;
  bitwise_xor(rhs: number): Column<Float64>;
  bitwise_xor<R extends Numeric>(rhs: Scalar<R>): Column<CommonType<T, R>>;
  bitwise_xor<R extends Numeric>(rhs: Column<R>): Column<CommonType<T, R>>;

  /**
   * Perform a binary `&&` operation between this Column and another Column or scalar value.
   *
   * @param rhs The other Column or scalar to use.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  logical_and(rhs: bigint): Column<Int64>;
  logical_and(rhs: number): Column<Float64>;
  logical_and<R extends Numeric>(rhs: Scalar<R>): Column<CommonType<T, R>>;
  logical_and<R extends Numeric>(rhs: Column<R>): Column<CommonType<T, R>>;

  /**
   * Perform a binary `||` operation between this Column and another Column or scalar value.
   *
   * @param rhs The other Column or scalar to use.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  logical_or(rhs: bigint): Column<Int64>;
  logical_or(rhs: number): Column<Float64>;
  logical_or<R extends Numeric>(rhs: Scalar<R>): Column<CommonType<T, R>>;
  logical_or<R extends Numeric>(rhs: Column<R>): Column<CommonType<T, R>>;

  /**
   * Perform a binary `coalesce` operation between this Column and another Column or scalar value.
   *
   * @param rhs The other Column or scalar to use.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  coalesce(rhs: bigint): Column<Int64>;
  coalesce(rhs: number): Column<Float64>;
  coalesce<R extends Numeric>(rhs: Scalar<R>): Column<CommonType<T, R>>;
  coalesce<R extends Numeric>(rhs: Column<R>): Column<CommonType<T, R>>;

  /**
   * Perform a binary `<<` operation between this Column and another Column or scalar value.
   *
   * @param rhs The other Column or scalar to use.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  shift_left(rhs: bigint): Column<Int64>;
  shift_left(rhs: number): Column<Float64>;
  shift_left<R extends Numeric>(rhs: Scalar<R>): Column<CommonType<T, R>>;
  shift_left<R extends Numeric>(rhs: Column<R>): Column<CommonType<T, R>>;

  /**
   * Perform a binary `>>` operation between this Column and another Column or scalar
   * value.
   *
   * @param rhs The other Column or scalar to use.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  shift_right(rhs: bigint): Column<Int64>;
  shift_right(rhs: number): Column<Float64>;
  shift_right<R extends Numeric>(rhs: Scalar<R>): Column<CommonType<T, R>>;
  shift_right<R extends Numeric>(rhs: Column<R>): Column<CommonType<T, R>>;

  /**
   * Perform a binary `shift_right_unsigned` operation between this Column and another Column or
   * scalar value.
   *
   * @param rhs The other Column or scalar to use.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  shift_right_unsigned(rhs: bigint): Column<Int64>;
  shift_right_unsigned(rhs: number): Column<Float64>;
  shift_right_unsigned<R extends Numeric>(rhs: Scalar<R>): Column<CommonType<T, R>>;
  shift_right_unsigned<R extends Numeric>(rhs: Column<R>): Column<CommonType<T, R>>;

  /**
   * Perform a binary `log_base` operation between this Column and another Column or scalar value.
   *
   * @param rhs The other Column or scalar to use.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  log_base(rhs: bigint): Column<Int64>;
  log_base(rhs: number): Column<Float64>;
  log_base<R extends Numeric>(rhs: Scalar<R>): Column<CommonType<T, R>>;
  log_base<R extends Numeric>(rhs: Column<R>): Column<CommonType<T, R>>;

  /**
   * Perform a binary `atan2` operation between this Column and another Column or scalar value.
   *
   * @param rhs The other Column or scalar to use.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  atan2(rhs: bigint): Column<Int64>;
  atan2(rhs: number): Column<Float64>;
  atan2<R extends Numeric>(rhs: Scalar<R>): Column<CommonType<T, R>>;
  atan2<R extends Numeric>(rhs: Column<R>): Column<CommonType<T, R>>;

  /**
   * Perform a binary `null_equals` operation between this Column and another Column or scalar
   * value.
   *
   * @param rhs The other Column or scalar to use.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  null_equals(rhs: bigint): Column<Bool8>;
  null_equals(rhs: number): Column<Bool8>;
  null_equals<R extends Numeric>(rhs: Scalar<R>): Column<Bool8>;
  null_equals<R extends Numeric>(rhs: Column<R>): Column<Bool8>;

  /**
   * Perform a binary `null_max` operation between this Column and another Column or scalar value.
   *
   * @param rhs The other Column or scalar to use.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  null_max(rhs: bigint): Column<Int64>;
  null_max(rhs: number): Column<Float64>;
  null_max<R extends Numeric>(rhs: Scalar<R>): Column<CommonType<T, R>>;
  null_max<R extends Numeric>(rhs: Column<R>): Column<CommonType<T, R>>;

  /**
   * Perform a binary `null_min` operation between this Column and another Column or scalar value.
   *
   * @param rhs The other Column or scalar to use.
   * @returns A Column of a common numeric type with the results of the binary operation.
   */
  null_min(rhs: bigint): Column<Int64>;
  null_min(rhs: number): Column<Float64>;
  null_min<R extends Numeric>(rhs: Scalar<R>): Column<CommonType<T, R>>;
  null_min<R extends Numeric>(rhs: Column<R>): Column<CommonType<T, R>>;
}

// eslint-disable-next-line @typescript-eslint/no-redeclare
export const Column: ColumnConstructor = CUDF.Column;
