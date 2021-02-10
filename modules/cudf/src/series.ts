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
import {
  DataType as ArrowDataType,
  RecordBatchReader,
  Table as ArrowTable,
  Vector
} from 'apache-arrow';
import {VectorType} from 'apache-arrow/interfaces';

import {Column, ColumnProps} from './column';
import {fromArrow} from './column/from_arrow';
import {DataFrame} from './data_frame';
import {Table} from './table';
import {
  ArrowToCUDFType,
  Bool8,
  CUDFToArrowType,
  DataType,
  Float32,
  Float64,
  Int16,
  Int32,
  Int64,
  Int8,
  Integral,
  NullOrder,
  SeriesType,
  TypeId,
  Uint16,
  Uint32,
  Uint64,
  Uint8,
  Utf8String,
} from './types';

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
  static new<T extends DataType>(input: Column<T>): SeriesType<T>;
  static new<T extends DataType>(input: SeriesProps<T>): SeriesType<T>;
  static new<T extends ArrowDataType>(input: Vector<T>): SeriesType<ArrowToCUDFType<T>>;
  static new<T extends DataType>(input: SeriesProps<T>|Column<T>|Vector<CUDFToArrowType<T>>) {
    /* eslint-disable @typescript-eslint/no-use-before-define */
    const column = asColumn(input);
    switch (column.type.id) {
      case TypeId.INT8: return new Int8Series(column);
      case TypeId.INT16: return new Int16Series(column);
      case TypeId.INT32: return new Int32Series(column);
      case TypeId.INT64: return new Int64Series(column);
      case TypeId.UINT8: return new Uint8Series(column);
      case TypeId.UINT16: return new Uint16Series(column);
      case TypeId.UINT32: return new Uint32Series(column);
      case TypeId.UINT64: return new Uint64Series(column);
      case TypeId.FLOAT32: return new Float32Series(column);
      case TypeId.FLOAT64: return new Float64Series(column);
      case TypeId.BOOL8: return new Bool8Series(column);
      case TypeId.STRING: return new StringSeries(column);
      default: throw new Error('Unknown DataType');
    }
    /* eslint-enable @typescript-eslint/no-use-before-define */
  }

  /** @ignore */
  public readonly _col: Column<T>;

  protected constructor(input: SeriesProps<T>|Column<T>|Vector<CUDFToArrowType<T>>) {
    this._col = asColumn(input);
    // TODO: implement the DataType subclasses in C++
    this._type = asSubType(this._col.type);
  }

  /** @ignore */
  private readonly _type: T;

  /**
   * The data type of elements in the underlying data.
   */
  get type() { return this._type; }

  /**
   * The DeviceBuffer for for the validity bitmask in GPU memory.
   */
  get mask() { return this._col.mask; }

  /**
   * The number of elements in this Series.
   */
  get length() { return this._col.length; }

  /**
   * A boolean indicating whether a validity bitmask exists.
   */
  get nullable() { return this._col.nullable; }

  /**
   * Whether the Series contains null elements.
   */
  get hasNulls() { return this._col.hasNulls; }

  /**
   * The number of null elements in this Series.
   */
  get nullCount() { return this._col.nullCount; }

  /**
   * The number of child columns in this Series.
   */
  get numChildren() { return this._col.numChildren; }

  /**
   * Return a sub-selection of this Series using the specified integral indices.
   *
   * @param selection A Series of 8/16/32-bit signed or unsigned integer indices.
   */
  gather<R extends Integral>(selection: Series<R>): SeriesType<T> {
    return Series.new(this._col.gather(selection._col));
  }

  /**
   * Return a sub-selection of this Series using the specified boolean mask.
   *
   * @param mask A Series of boolean values for whose corresponding element in this Series will be
   *   selected or ignored.
   */
  filter(mask: Series<Bool8>): SeriesType<T> { return Series.new(this._col.gather(mask._col)); }

  /**
   * Return a child at the specified index to host memory
   *
   * @param index
   */
  getChild<R extends DataType = any>(index: number): SeriesType<T> {
    return Series.new(this._col.getChild<R>(index));
  }

  /**
   * Return a value at the specified index to host memory
   *
   * @param index
   */
  getValue(index: number) { return this._col.getValue(index); }

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
  setNullMask(mask: DeviceBuffer, nullCount?: number) { this._col.setNullMask(mask, nullCount); }

  /**
   * Copy a Series to an Arrow vector in host memory
   */
  toArrow(): VectorType<CUDFToArrowType<T>> {
    const reader = RecordBatchReader.from(new Table({columns: [this._col]}).toArrow([[0]]));
    const column = new ArrowTable(reader.schema, [...reader]).getColumnAt<CUDFToArrowType<T>>(0);
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    return column!.chunks[0] as VectorType<CUDFToArrowType<T>>;
  }

  /**
   * Generate an ordering that sorts the Series in a specified way.
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
   * Generate a new Series that is sorted in a specified way.
   *
   * @param ascending whether to sort ascending (true) or descending (false)
   *   Default: true
   * @param null_order whether nulls should sort before or after other values
   *   Default: BEFORE
   *
   * @returns Sorted values
   */
  sortValues(ascending = true, null_order: NullOrder = NullOrder.BEFORE): SeriesType<T> {
    return this.gather(this.orderBy(ascending, null_order));
  }

  /**
   * Creates a Series of `BOOL8` elements where `true` indicates the value is null and `false`
   * indicates the value is valid.
   *
   * @param memoryResource Memory resource used to allocate the result Column's device memory.
   * @returns A non-nullable Series of `BOOL8` elements with `true` representing `null`
   *   values.
   */
  isNull(memoryResource?: MemoryResource) { return Series.new(this._col.isNull(memoryResource)); }

  /**
   * Creates a Series of `BOOL8` elements where `true` indicates the value is valid and `false`
   * indicates the value is null.
   *
   * @param memoryResource Memory resource used to allocate the result Column's device memory.
   * @returns A non-nullable Series of `BOOL8` elements with `false` representing `null`
   *   values.
   */
  isValid(memoryResource?: MemoryResource) { return Series.new(this._col.isValid(memoryResource)); }

  /**
   * drop Null values from the series
   *
   * @returns series without Null values
   */
  dropNA(): Series<T>{return Series.new(this._col.drop_nulls())}

  /**
   * Return whether all elements are true in Series.
   *
   * @param skipna bool
   * Exclude null values. If the entire row/column is NA and skipna is true, then the result will
   * be true, as for an empty row/column. If skipna is false, then NA are treated as true, because
   * these are not equal to zero.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   *
   * @returns true if all elements are true in Series, else false.
   */
  all(skipna = true, memoryResource?: MemoryResource): boolean {
    if (skipna) {
      if (this?.length == this?.nullCount) { return true; }
    }
    return Boolean(this._col.all(memoryResource));
  }

  /**
   * Return whether any elements are true in Series.
   *
   * @param skipna bool
   * Exclude NA/null values. If the entire row/column is NA and skipna is true, then the result will
   * be true, as for an empty row/column. If skipna is false, then NA are treated as true, because
   * these are not equal to zero.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   *
   * @returns true if any elements are true in Series, else false.
   */
  any(skipna = true, memoryResource?: MemoryResource): boolean {
    if (this.length == 0) { return false; }
    if (skipna) {
      if (this?.length == this?.nullCount) { return false; }
    }
    return Boolean(this._col.any(memoryResource));
  }
}

import {Bool8Series} from './series/bool';
import {Float32Series, Float64Series} from './series/float';
import {
  Int8Series,
  Int16Series,
  Int32Series,
  Uint8Series,
  Uint16Series,
  Uint32Series,
  Int64Series,
  Uint64Series
} from './series/integral';
import {StringSeries} from './series/string';

function asColumn<T extends DataType>(value: SeriesProps<T>|Column<T>|Vector<CUDFToArrowType<T>>) {
  if (value instanceof Column) {
    return value;
  } else if (value instanceof Vector) {
    return fromArrow(value);
  } else {
    const props: ColumnProps<T> = {
      type: value.type.id,
      data: value.data,
      offset: value.offset,
      length: value.length,
      nullCount: value.nullCount,
      nullMask: value.nullMask,
    };
    if (value.children != null) {
      props.children = value.children.map((item: Series) => item._col);
    }
    return new Column(props);
  }
}

function asSubType<T extends DataType>(type: T): T {
  switch (type.id) {
    case TypeId.INT8: return (type instanceof Int8 ? type : new Int8) as T;
    case TypeId.INT16: return (type instanceof Int16 ? type : new Int16) as T;
    case TypeId.INT32: return (type instanceof Int32 ? type : new Int32) as T;
    case TypeId.INT64: return (type instanceof Int64 ? type : new Int64) as T;
    case TypeId.UINT8: return (type instanceof Uint8 ? type : new Uint8) as T;
    case TypeId.UINT16: return (type instanceof Uint16 ? type : new Uint16) as T;
    case TypeId.UINT32: return (type instanceof Uint32 ? type : new Uint32) as T;
    case TypeId.UINT64: return (type instanceof Uint64 ? type : new Uint64) as T;
    case TypeId.FLOAT32: return (type instanceof Float32 ? type : new Float32) as T;
    case TypeId.FLOAT64: return (type instanceof Float64 ? type : new Float64) as T;
    case TypeId.BOOL8: return (type instanceof Bool8 ? type : new Bool8) as T;
    case TypeId.STRING: return (type instanceof Utf8String ? type : new Utf8String) as T;
    default: throw new Error(`Unknown TypeId "${TypeId[type.id]}"`);
  }
}

export {
  Bool8Series,
  Float32Series,
  Float64Series,
  Int8Series,
  Int16Series,
  Int32Series,
  Uint8Series,
  Uint16Series,
  Uint32Series,
  Int64Series,
  Uint64Series,
  StringSeries
};
