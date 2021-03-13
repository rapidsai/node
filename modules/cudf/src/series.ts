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
import {DeviceBuffer, MemoryResource} from '@rapidsai/rmm';
import * as arrow from 'apache-arrow';
import {VectorType} from 'apache-arrow/interfaces';

import {Column, ColumnProps} from './column';
import {fromArrow} from './column/from_arrow';
import {Table, ToArrowMetadata} from './table';
import {
  Bool8,
  DataType,
  Float32,
  Float64,
  IndexType,
  Int16,
  Int32,
  Int64,
  Int8,
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
  /*
   * SeriesProps *with* a `nullMask` shouldn't allow `data` to be an Array with elements and nulls:
   * ```javascript
   * Series.new({
   *   type: new Int32,
   *   data: [1, 0, 2, 3, 0], ///< must not include nulls
   *   nullMask: [true, false, true, true, false]
   * })
   *  ```
   */
  type: T;
  data?: DeviceBuffer | MemoryData | T['scalarType'][] | null;
  offset?: number;
  length?: number;
  nullCount?: number;
  nullMask?: DeviceBuffer | MemoryData | any[] | boolean | null;
  children?: ReadonlyArray<Series>| null;
}|{
  /*
   * SeriesProps *without* a `nullMask` should allow `data` to be an Array with elements and nulls:
   * ```javascript
   * Series.new({
   *   type: new Int32,
   *   data: [1, null, 2, 3, null] ///< can include nulls
   * })
   *  ```
   */
  type: T;
  data?: DeviceBuffer|MemoryData|(T['scalarType'] | null | undefined)[]|null;
  offset?: number;
  length?: number;
  nullCount?: number;
  nullMask?: never;
  children?: ReadonlyArray<Series>|null;
};

export type Series<T extends arrow.DataType = any> = {
  [arrow.Type.NONE]: never,  // TODO
  [arrow.Type.Null]: never,  // TODO
  [arrow.Type.Int]: never,
  [arrow.Type.Int8]: Int8Series,
  [arrow.Type.Int16]: Int16Series,
  [arrow.Type.Int32]: Int32Series,
  [arrow.Type.Int64]: Int64Series,
  [arrow.Type.Uint8]: Uint8Series,
  [arrow.Type.Uint16]: Uint16Series,
  [arrow.Type.Uint32]: Uint32Series,
  [arrow.Type.Uint64]: Uint64Series,
  [arrow.Type.Float]: never,
  [arrow.Type.Float16]: never,
  [arrow.Type.Float32]: Float32Series,
  [arrow.Type.Float64]: Float64Series,
  [arrow.Type.Binary]: never,
  [arrow.Type.Utf8]: StringSeries,
  [arrow.Type.Bool]: Bool8Series,
  [arrow.Type.Decimal]: never,               // TODO
  [arrow.Type.Date]: never,                  // TODO
  [arrow.Type.DateDay]: never,               // TODO
  [arrow.Type.DateMillisecond]: never,       // TODO
  [arrow.Type.Time]: never,                  // TODO
  [arrow.Type.TimeSecond]: never,            // TODO
  [arrow.Type.TimeMillisecond]: never,       // TODO
  [arrow.Type.TimeMicrosecond]: never,       // TODO
  [arrow.Type.TimeNanosecond]: never,        // TODO
  [arrow.Type.Timestamp]: never,             // TODO
  [arrow.Type.TimestampSecond]: never,       // TODO
  [arrow.Type.TimestampMillisecond]: never,  // TODO
  [arrow.Type.TimestampMicrosecond]: never,  // TODO
  [arrow.Type.TimestampNanosecond]: never,   // TODO
  [arrow.Type.Interval]: never,              // TODO
  [arrow.Type.IntervalDayTime]: never,       // TODO
  [arrow.Type.IntervalYearMonth]: never,     // TODO
  [arrow.Type.List]: ListSeries<(T extends List ? T['childType'] : any)>,
  [arrow.Type.Struct]: StructSeries<(T extends Struct ? T['childTypes'] : any)>,
  [arrow.Type.Union]: never,            // TODO
  [arrow.Type.DenseUnion]: never,       // TODO
  [arrow.Type.SparseUnion]: never,      // TODO
  [arrow.Type.FixedSizeBinary]: never,  // TODO
  [arrow.Type.FixedSizeList]: never,    // TODO
  [arrow.Type.Map]: never,              // TODO
  [arrow.Type.Dictionary]: never,       // TODO
}[T['TType']];

/**
 * One-dimensional GPU array
 */
export class AbstractSeries<T extends DataType = any> {
  static new<T extends arrow.Vector>(input: T): Series<ArrowToCUDFType<T['type']>>;
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
   * The offset of elements in this Series underlying Column.
   */
  get offset() { return this._col.offset; }

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
  gather<R extends IndexType>(selection: Series<R>): Series<T> {
    return this.__construct(this._col.gather(selection._col));
  }

  /**
   * Return a sub-selection of this Series using the specified boolean mask.
   *
   * @param mask A Series of boolean values for whose corresponding element in this Series will be
   *   selected or ignored.
   */
  filter(mask: Series<Bool8>): Series<T> { return this.__construct(this._col.gather(mask._col)); }

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
    const names = (name: string|number, type?: DataType): ToArrowMetadata => {
      if (!type || !type.children || !type.children.length) { return [name]; }
      if (type instanceof arrow.List) {
        if (!type.children[0]) { return [name, [[0], [1]]]; }
        return [name, [[0], names(type.children[0].name, type.children[0].type)]];
      }
      return [name, type.children.map((f) => names(f.name, f.type))];
    };
    const reader = arrow.RecordBatchReader.from(
      new Table({columns: [this._col]}).toArrow([names(0, this.type)]));
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

  /**
   * drop Null values from the series
   * @param memoryResource Memory resource used to allocate the result Column's device memory.
   * @returns series without Null values
   */
  dropNulls(memoryResource?: MemoryResource): Series<T> {
    return this.__construct(this._col.drop_nulls(memoryResource));
  }

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
  all(skipna = true, memoryResource?: MemoryResource) {
    if (skipna) {
      if (this.length == this.nullCount) { return true; }
    }
    return this._col.all(memoryResource);
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
  any(skipna = true, memoryResource?: MemoryResource) {
    if (this.length == 0) { return false; }
    if (skipna) {
      if (this.length == this.nullCount) { return false; }
    }
    return this._col.any(memoryResource);
  }
  /**
   * @summary Hook for specialized Series to override when constructing from a C++ Column.
   */
  protected __construct(inp: Column<T>): Series<T> { return Series.new(inp); }
}

// eslint-disable-next-line @typescript-eslint/no-redeclare
export const Series = AbstractSeries;

Object.defineProperty(Series.prototype, '__construct', {
  writable: false,
  enumerable: false,
  configurable: true,
  value: (Series.prototype as any).__construct,
});

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
    const props: ColumnProps<T> = {...value};
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
    // public visitNull                 <T extends Null>(col: Column<T>) { return new (NullSeries as any)(col); }
    public visitBool                 <T extends Bool8>(col: Column<T>) { return new (Bool8Series as any)(col); }
    public visitInt8                 <T extends Int8>(col: Column<T>) { return new (Int8Series as any)(col); }
    public visitInt16                <T extends Int16>(col: Column<T>) { return new (Int16Series as any)(col); }
    public visitInt32                <T extends Int32>(col: Column<T>) { return new (Int32Series as any)(col); }
    public visitInt64                <T extends Int64>(col: Column<T>) { return new (Int64Series as any)(col); }
    public visitUint8                <T extends Uint8>(col: Column<T>) { return new (Uint8Series as any)(col); }
    public visitUint16               <T extends Uint16>(col: Column<T>) { return new (Uint16Series as any)(col); }
    public visitUint32               <T extends Uint32>(col: Column<T>) { return new (Uint32Series as any)(col); }
    public visitUint64               <T extends Uint64>(col: Column<T>) { return new (Uint64Series as any)(col); }
    // public visitFloat16              <T extends Float16>(_: T) { return new (Float16Series as any)(_); }
    public visitFloat32              <T extends Float32>(col: Column<T>) { return new (Float32Series as any)(col); }
    public visitFloat64              <T extends Float64>(col: Column<T>) { return new (Float64Series as any)(col); }
    public visitUtf8                 <T extends Utf8String>(col: Column<T>) { return new (StringSeries as any)(col); }
    // public visitBinary               <T extends Binary>(col: Column<T>) { return new (BinarySeries as any)(col); }
    // public visitFixedSizeBinary      <T extends FixedSizeBinary>(col: Column<T>) { return new (FixedSizeBinarySeries as any)(col); }
    // public visitDateDay              <T extends DateDay>(col: Column<T>) { return new (DateDaySeries as any)(col); }
    // public visitDateMillisecond      <T extends DateMillisecond>(col: Column<T>) { return new (DateMillisecondSeries as any)(col); }
    // public visitTimestampSecond      <T extends TimestampSecond>(col: Column<T>) { return new (TimestampSecondSeries as any)(col); }
    // public visitTimestampMillisecond <T extends TimestampMillisecond>(col: Column<T>) { return new (TimestampMillisecondSeries as any)(col); }
    // public visitTimestampMicrosecond <T extends TimestampMicrosecond>(col: Column<T>) { return new (TimestampMicrosecondSeries as any)(col); }
    // public visitTimestampNanosecond  <T extends TimestampNanosecond>(col: Column<T>) { return new (TimestampNanosecondSeries as any)(col); }
    // public visitTimeSecond           <T extends TimeSecond>(col: Column<T>) { return new (TimeSecondSeries as any)(col); }
    // public visitTimeMillisecond      <T extends TimeMillisecond>(col: Column<T>) { return new (TimeMillisecondSeries as any)(col); }
    // public visitTimeMicrosecond      <T extends TimeMicrosecond>(col: Column<T>) { return new (TimeMicrosecondSeries as any)(col); }
    // public visitTimeNanosecond       <T extends TimeNanosecond>(col: Column<T>) { return new (TimeNanosecondSeries as any)(col); }
    // public visitDecimal              <T extends Decimal>(col: Column<T>) { return new (DecimalSeries as any)(col); }
    public visitList                 <T extends List>(col: Column<T>) { return new (ListSeries as any)(col); }
    public visitStruct               <T extends Struct>(col: Column<T>) { return new (StructSeries as any)(col); }
    // public visitDenseUnion           <T extends DenseUnion>(col: Column<T>) { return new (DenseUnionSeries as any)(col); }
    // public visitSparseUnion          <T extends SparseUnion>(col: Column<T>) { return new (SparseUnionSeries as any)(col); }
    // public visitDictionary           <T extends Dictionary>(col: Column<T>) { return new (DictionarySeries as any)(col); }
    // public visitIntervalDayTime      <T extends IntervalDayTime>(col: Column<T>) { return new (IntervalDayTimeSeries as any)(col); }
    // public visitIntervalYearMonth    <T extends IntervalYearMonth>(col: Column<T>) { return new (IntervalYearMonthSeries as any)(col); }
    // public visitFixedSizeList        <T extends FixedSizeList>(col: Column<T>) { return new (FixedSizeListSeries as any)(col); }
    // public visitMap                  <T extends Map>(col: Column<T>) { return new (MapSeries as any)(col); }
  }
  /* eslint-enable @typescript-eslint/no-unused-vars */
  // clang-format on
  const visitor = new ColumnToSeriesVisitor();
  return function columnToSeries<T extends DataType>(column: Column<T>) {
    return visitor.visit(column);
  };
})();
