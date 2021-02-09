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
import * as arrow from 'apache-arrow';
import {VectorType} from 'apache-arrow/interfaces';

import {Column, ColumnProps} from './column';
import {fromArrow} from './column/from_arrow';
import {Table} from './table';
import {
  Bool8,
  DataType,
  Float32,
  Float64,
  Int16,
  Int32,
  Int64,
  Int8,
  Integral,
  List,
  Struct,
  Uint16,
  Uint32,
  Uint64,
  Uint8,
  Utf8String,
} from './types/dtypes';
import {
  NullOrder,
} from './types/enums';
import {ArrowToCUDFType, arrowToCUDFType} from './types/mappings';

export type SeriesProps<T extends DataType = any> = {
  type: T,
  data?: DeviceBuffer|MemoryData|null,
  offset?: number,
  length?: number,
  nullCount?: number,
  nullMask?: DeviceBuffer|MemoryData|null,
  children?: ReadonlyArray<Series>|null
};

export type Series<T extends DataType = any> = {
  [key: number]: SeriesBase,
  [arrow.Type.NONE]: never,  // TODO
  [arrow.Type.Null]: never,  // TODO
  [arrow.Type.Int8]: Int8Series,
  [arrow.Type.Int16]: Int16Series,
  [arrow.Type.Int32]: Int32Series,
  [arrow.Type.Int64]: Int64Series,
  [arrow.Type.Uint8]: Uint8Series,
  [arrow.Type.Uint16]: Uint16Series,
  [arrow.Type.Uint32]: Uint32Series,
  [arrow.Type.Uint64]: Uint64Series,
  [arrow.Type.Float32]: Float32Series,
  [arrow.Type.Float64]: Float64Series,
  [arrow.Type.Bool]: Bool8Series,
  [arrow.Type.DateDay]: never,               // TODO
  [arrow.Type.DateMillisecond]: never,       // TODO
  [arrow.Type.TimestampSecond]: never,       // TODO
  [arrow.Type.TimestampMillisecond]: never,  // TODO
  [arrow.Type.TimestampMicrosecond]: never,  // TODO
  [arrow.Type.TimestampNanosecond]: never,   // TODO
  // [arrow.Type.DURATION_DAYS]: never,           // TODO
  // [arrow.Type.DURATION_SECONDS]: never,        // TODO
  // [arrow.Type.DURATION_MILLISECONDS]: never,   // TODO
  // [arrow.Type.DURATION_MICROSECONDS]: never,   // TODO
  // [arrow.Type.DURATION_NANOSECONDS]: never,    // TODO
  [arrow.Type.Dictionary]: never,  // TODO
  [arrow.Type.Utf8]: StringSeries,
  [arrow.Type.List]: ListSeries<ArrowToCUDFType<T extends arrow.List ? T : any>['childType']>,
  // [arrow.Type.DECIMAL32]: never,  // TODO
  // [arrow.Type.DECIMAL64]: never,  // TODO
  [arrow.Type.Struct]:
    StructSeries<ArrowToCUDFType<T extends arrow.Struct ? T : any>['childTypes']>,
}[T['typeId']];

/**
 * One-dimensional GPU array
 */
class SeriesBase<T extends DataType = any> {
  static new<T extends arrow.Vector>(input: T): Series<T['type']>;
  static new<T extends DataType>(input: Column<T>|SeriesProps<T>): Series<T>;
  static new<T extends DataType>(input: Column<T>|SeriesProps<T>|arrow.Vector<T>) {
    return columnToSeries(asColumn<T>(input)) as any as Series<T>;
  }

  /** @ignore */
  public readonly _col: Column<T>;

  protected constructor(input: SeriesProps<T>|Column<T>|arrow.Vector<T>) {
    this._col = asColumn<T>(input);
  }

  /**
   * The data type of elements in the underlying data.
   */
  get type() { return this._col.type; }

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
  gather<R extends Integral>(selection: Series<R>): Series<T> {
    return Series.new(this._col.gather(selection._col));
  }

  /**
   * Return a sub-selection of this Series using the specified boolean mask.
   *
   * @param mask A Series of boolean values for whose corresponding element in this Series will be
   *   selected or ignored.
   */
  filter(mask: Series<Bool8>): Series<T> { return Series.new(this._col.gather(mask._col)); }

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
  [Symbol.iterator](): IterableIterator<T['scalarType']|null> {
    return this.toArrow()[Symbol.iterator]();
  }

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
  toArrow(): VectorType<T> {
    const reader = arrow.RecordBatchReader.from(new Table({columns: [this._col]}).toArrow([[0]]));
    const column = new arrow.Table(reader.schema, [...reader]).getColumnAt<T>(0);
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    return column!.chunks[0] as VectorType<T>;
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
    return Series.new(new Table({columns: [this._col]}).orderBy([ascending], [null_order]));
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
  sortValues(ascending = true, null_order: NullOrder = NullOrder.BEFORE): Series<T> {
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
}

// eslint-disable-next-line @typescript-eslint/no-redeclare
export const Series = SeriesBase;

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
import {ListSeries} from './series/list';
import {StructSeries} from './series/struct';

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
  StringSeries,
  ListSeries,
  StructSeries,
};

function asColumn<T extends DataType>(value: SeriesProps<T>|Column<T>|arrow.Vector<T>): Column<T> {
  if (value instanceof arrow.Vector) { return fromArrow(value) as any; }
  if (!(value.type instanceof arrow.DataType)) {
    (value as any).type = arrowToCUDFType<T>(value.type);
  }
  if (value instanceof Column) {
    return value;
  } else {
    const props: ColumnProps<T> = {
      type: value.type,
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

const columnToSeries = (() => {
  interface ColumnToSeriesVisitor extends arrow.Visitor {
    visit<T extends DataType>(column: Column<T>): Series<T>;
    visitMany<T extends DataType>(columns: Column<T>[]): Series<T>[];
    getVisitFn<T extends DataType>(column: Column<T>): (column: Column<T>) => Series<T>;
  }
  // clang-format off
  /* eslint-disable @typescript-eslint/no-unused-vars */
  class ColumnToSeriesVisitor extends arrow.Visitor {
    getVisitFn<T extends DataType>(column: Column<T>): (column: Column<T>) => Series<T> {
      if (!(column.type instanceof arrow.DataType)) {
        return super.getVisitFn({
          ...(column.type as any),
          __proto__: arrow.DataType.prototype
        });
      }
      return super.getVisitFn(column.type);
    }
    // public visitNull                 <T extends Null>(_: Column<T>) { return new (NullSeries as any)(_); }
    public visitBool                 <T extends Bool8>(_: Column<T>) { return new (Bool8Series as any)(_); }
    public visitInt8                 <T extends Int8>(_: Column<T>) { return new (Int8Series as any)(_); }
    public visitInt16                <T extends Int16>(_: Column<T>) { return new (Int16Series as any)(_); }
    public visitInt32                <T extends Int32>(_: Column<T>) { return new (Int32Series as any)(_); }
    public visitInt64                <T extends Int64>(_: Column<T>) { return new (Int64Series as any)(_); }
    public visitUint8                <T extends Uint8>(_: Column<T>) { return new (Uint8Series as any)(_); }
    public visitUint16               <T extends Uint16>(_: Column<T>) { return new (Uint16Series as any)(_); }
    public visitUint32               <T extends Uint32>(_: Column<T>) { return new (Uint32Series as any)(_); }
    public visitUint64               <T extends Uint64>(_: Column<T>) { return new (Uint64Series as any)(_); }
    // public visitFloat16              <T extends Float16>(_: T) { return new (Float16Series as any)(_); }
    public visitFloat32              <T extends Float32>(_: Column<T>) { return new (Float32Series as any)(_); }
    public visitFloat64              <T extends Float64>(_: Column<T>) { return new (Float64Series as any)(_); }
    public visitUtf8                 <T extends Utf8String>(_: Column<T>) { return new (StringSeries as any)(_); }
    // public visitBinary               <T extends Binary>(_: Column<T>) { return new (BinarySeries as any)(_); }
    // public visitFixedSizeBinary      <T extends FixedSizeBinary>(_: Column<T>) { return new (FixedSizeBinarySeries as any)(_); }
    // public visitDateDay              <T extends DateDay>(_: Column<T>) { return new (DateDaySeries as any)(_); }
    // public visitDateMillisecond      <T extends DateMillisecond>(_: Column<T>) { return new (DateMillisecondSeries as any)(_); }
    // public visitTimestampSecond      <T extends TimestampSecond>(_: Column<T>) { return new (TimestampSecondSeries as any)(_); }
    // public visitTimestampMillisecond <T extends TimestampMillisecond>(_: Column<T>) { return new (TimestampMillisecondSeries as any)(_); }
    // public visitTimestampMicrosecond <T extends TimestampMicrosecond>(_: Column<T>) { return new (TimestampMicrosecondSeries as any)(_); }
    // public visitTimestampNanosecond  <T extends TimestampNanosecond>(_: Column<T>) { return new (TimestampNanosecondSeries as any)(_); }
    // public visitTimeSecond           <T extends TimeSecond>(_: Column<T>) { return new (TimeSecondSeries as any)(_); }
    // public visitTimeMillisecond      <T extends TimeMillisecond>(_: Column<T>) { return new (TimeMillisecondSeries as any)(_); }
    // public visitTimeMicrosecond      <T extends TimeMicrosecond>(_: Column<T>) { return new (TimeMicrosecondSeries as any)(_); }
    // public visitTimeNanosecond       <T extends TimeNanosecond>(_: Column<T>) { return new (TimeNanosecondSeries as any)(_); }
    // public visitDecimal              <T extends Decimal>(_: Column<T>) { return new (DecimalSeries as any)(_); }
    public visitList                 <T extends List>(_: Column<T>) { return new (ListSeries as any)(_); }
    public visitStruct               <T extends Struct>(_: Column<T>) { return new (StructSeries as any)(_); }
    // public visitDenseUnion           <T extends DenseUnion>(_: Column<T>) { return new (DenseUnionSeries as any)(_); }
    // public visitSparseUnion          <T extends SparseUnion>(_: Column<T>) { return new (SparseUnionSeries as any)(_); }
    // public visitDictionary           <T extends Dictionary>(_: Column<T>) { return new (DictionarySeries as any)(_); }
    // public visitIntervalDayTime      <T extends IntervalDayTime>(_: Column<T>) { return new (IntervalDayTimeSeries as any)(_); }
    // public visitIntervalYearMonth    <T extends IntervalYearMonth>(_: Column<T>) { return new (IntervalYearMonthSeries as any)(_); }
    // public visitFixedSizeList        <T extends FixedSizeList>(_: Column<T>) { return new (FixedSizeListSeries as any)(_); }
    // public visitMap                  <T extends Map>(_: Column<T>) { return new (MapSeries as any)(_); }
  }
  /* eslint-enable @typescript-eslint/no-unused-vars */
  // clang-format on
  const visitor = new ColumnToSeriesVisitor();
  return function columnToSeries<T extends DataType>(column: Column<T>) {
    return visitor.visit(column);
  };
})();
