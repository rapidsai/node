// Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
  Float32Buffer,
  Float64Buffer,
  Int16Buffer,
  Int32Buffer,
  Int64Buffer,
  Int8Buffer,
  MemoryData,
  MemoryView,
  Uint16Buffer,
  Uint32Buffer,
  Uint64Buffer,
  Uint8Buffer,
  Uint8ClampedBuffer,
} from '@rapidsai/cuda';
import {DeviceBuffer, MemoryResource} from '@rapidsai/rmm';
import * as arrow from 'apache-arrow';
import {JavaScriptArrayDataType} from 'apache-arrow/interfaces';
import {compareTypes} from 'apache-arrow/visitor/typecomparator';

import {Column} from './column';
import {fromArrow} from './column/from_arrow';
import {DataFrame} from './data_frame';
import {DisplayOptions} from './dataframe/print';
import {Scalar} from './scalar';
import {DISPOSER, scope} from './scope';
import {Table} from './table';
import {
  Bool8,
  Categorical,
  DataType,
  Float32,
  Float64,
  IndexType,
  Int16,
  Int32,
  Int64,
  Int8,
  Integral,
  List,
  Numeric,
  Struct,
  TimestampDay,
  TimestampMicrosecond,
  TimestampMillisecond,
  TimestampNanosecond,
  TimestampSecond,
  Uint16,
  Uint32,
  Uint64,
  Uint8,
  Utf8String,
} from './types/dtypes';
import {
  DuplicateKeepOption,
  NullOrder,
} from './types/enums';
import {ArrowToCUDFType, CommonType, findCommonType} from './types/mappings';

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
  data?: DeviceBuffer | MemoryData | arrow.Vector<T>| T['scalarType'][] | null;
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
  data?: DeviceBuffer|MemoryData|arrow.Vector<T>|(T['scalarType'] | null | undefined)[]|null;
  offset?: number;
  length?: number;
  nullCount?: number;
  nullMask?: never;
  children?: ReadonlyArray<Series>|null;
};

// clang-format off
/* eslint-disable @typescript-eslint/no-unused-vars */
class CastVisitor<T extends DataType> extends arrow.Visitor {
  constructor(private series: AbstractSeries<T>, private memoryResource?: MemoryResource) { super(); }

  public visitBool                 <T extends Bool8>(_dtype: T) { return this.series._castAsBool8(this.memoryResource); }
  public visitInt8                 <T extends Int8>(_dtype: T) { return this.series._castAsInt8(this.memoryResource); }
  public visitInt16                <T extends Int16>(_dtype: T) { return this.series._castAsInt16(this.memoryResource); }
  public visitInt32                <T extends Int32>(_dtype: T) { return this.series._castAsInt32(this.memoryResource); }
  public visitInt64                <T extends Int64>(_dtype: T) { return this.series._castAsInt64(this.memoryResource); }
  public visitUint8                <T extends Uint8>(_dtype: T) { return this.series._castAsUint8(this.memoryResource); }
  public visitUint16               <T extends Uint16>(_dtype: T) { return this.series._castAsUint16(this.memoryResource); }
  public visitUint32               <T extends Uint32>(_dtype: T) { return this.series._castAsUint32(this.memoryResource); }
  public visitUint64               <T extends Uint64>(_dtype: T) { return this.series._castAsUint64(this.memoryResource); }
  public visitFloat32              <T extends Float32>(_dtype: T) { return this.series._castAsFloat32(this.memoryResource); }
  public visitFloat64              <T extends Float64>(_dtype: T) { return this.series._castAsFloat64(this.memoryResource); }
  public visitUtf8                 <T extends Utf8String>(_dtype: T) { return this.series._castAsString(this.memoryResource); }
  public visitDateDay              <T extends TimestampDay>(_dtype: T) { return this.series._castAsTimeStampDay(this.memoryResource); }
  public visitDateMillisecond      <T extends TimestampMillisecond>(_dtype: T) { return this.series._castAsTimeStampMillisecond(this.memoryResource); }
  public visitTimestampSecond      <T extends TimestampSecond>(_dtype: T) { return this.series._castAsTimeStampSecond(this.memoryResource); }
  public visitTimestampMillisecond <T extends TimestampMillisecond>(_dtype: T) { return this.series._castAsTimeStampMillisecond(this.memoryResource); }
  public visitTimestampMicrosecond <T extends TimestampMicrosecond>(_dtype: T) { return this.series._castAsTimeStampMicrosecond(this.memoryResource); }
  public visitTimestampNanosecond  <T extends TimestampNanosecond>(_dtype: T) { return this.series._castAsTimeStampNanosecond(this.memoryResource); }
  public visitDictionary           <T extends Categorical>(dtype: T) { return this.series._castAsCategorical(dtype, this.memoryResource); }
}
/* eslint-enable @typescript-eslint/no-unused-vars */
// clang-format on

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
  [arrow.Type.Decimal]: never,  // TODO
  [arrow.Type.Date]: never,     // TODO
  [arrow.Type.DateDay]: TimestampDaySeries,
  [arrow.Type.DateMillisecond]: TimestampMillisecondSeries,
  [arrow.Type.Time]: never,             // TODO
  [arrow.Type.TimeSecond]: never,       // TODO
  [arrow.Type.TimeMillisecond]: never,  // TODO
  [arrow.Type.TimeMicrosecond]: never,  // TODO
  [arrow.Type.TimeNanosecond]: never,   // TODO
  [arrow.Type.Timestamp]: never,        // TODO
  [arrow.Type.TimestampSecond]: TimestampSecondSeries,
  [arrow.Type.TimestampMillisecond]: TimestampMillisecondSeries,
  [arrow.Type.TimestampMicrosecond]: TimestampMicrosecondSeries,
  [arrow.Type.TimestampNanosecond]: TimestampNanosecondSeries,
  [arrow.Type.Interval]: never,           // TODO
  [arrow.Type.IntervalDayTime]: never,    // TODO
  [arrow.Type.IntervalYearMonth]: never,  // TODO
  [arrow.Type.List]: ListSeries<(T extends List ? T['valueType'] : any)>,
  [arrow.Type.Struct]: StructSeries<(T extends Struct ? T['dataTypes'] : any)>,
  [arrow.Type.Union]: never,            // TODO
  [arrow.Type.DenseUnion]: never,       // TODO
  [arrow.Type.SparseUnion]: never,      // TODO
  [arrow.Type.FixedSizeBinary]: never,  // TODO
  [arrow.Type.FixedSizeList]: never,    // TODO
  [arrow.Type.Map]: never,              // TODO
  [arrow.Type.Dictionary]: CategoricalSeries<(T extends arrow.Dictionary ? T['valueType'] : any)>
}[T['TType']];

/**
 * One-dimensional GPU array
 */
export class AbstractSeries<T extends DataType = any> {
  /**
   * Create a new cudf.Series from an apache arrow vector
   *
   * @example
   * ```typescript
   * import {Series, Int32} from '@rapidsai/cudf';
   * import * as arrow from 'apache-arrow';
   *
   * const arrow_vec = arrow.vectorFromArray(new Int32Array([1,2,3,4])));
   * const a = Series.new(arrow_vec); // Int32Series [1, 2, 3, 4]
   *
   * const arrow_vec_list = arrow.vectorFromArray(
   *   [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
   *   new arrow.List(arrow.Field.new({ name: 'ints', type: new arrow.Int32 })),
   * );
   *
   * const b = Series.new(arrow_vec_list) // ListSeries [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
   *
   * const arrow_vec_struct = arrow.vectorFromArray(
   *   [{ x: 0, y: 3 }, { x: 1, y: 4 }, { x: 2, y: 5 }],
   *   new arrow.Struct([
   *     arrow.Field.new({ name: 'x', type: new arrow.Int32 }),
   *     arrow.Field.new({ name: 'y', type: new arrow.Int32 })
   *   ]),
   * );
   *
   * const c = Series.new(arrow_vec_struct);
   * // StructSeries  [{ x: 0, y: 3 }, { x: 1, y: 4 }, { x: 2, y: 5 }]
   * ```
   */
  static new<T extends arrow.Vector>(input: T): Series<ArrowToCUDFType<T['type']>>;
  /**
   * Create a new cudf.Series from SeriesProps or a cudf.Column
   *
   * @example
   * ```typescript
   * import {Series, Int32} from '@rapidsai/cudf';
   *
   * //using SeriesProps
   * const a = Series.new({type: new Int32, data: [1, 2, 3, 4]}); // Int32Series [1, 2, 3, 4]
   *
   * //using underlying cudf.Column
   * const b = Series.new(a._col); // Int32Series [1, 2, 3, 4]
   * ```
   */
  static new<T extends AbstractSeries>(input: T): T;
  static new<T extends DataType>(input: Column<T>): Series<T>;
  static new<T extends DataType>(input: SeriesProps<T>): Series<T>;

  /**
   * Create a new cudf.Int8Series
   *
   * @example
   * ```typescript
   * import {
   *  Series,
   *  Int8Series,
   *  Int8
   * } from '@rapidsai/cudf';
   *
   * // Int8Series [1, 2, 3]
   * const a = Series.new(new Int8Array([1, 2, 3]));
   * const b = Series.new(new Int8Buffer([1, 2, 3]));
   * ```
   */
  static new(input: Int8Array|Int8Buffer): Series<Int8>;

  /**
   * Create a new cudf.Int16Series
   *
   * @example
   * ```typescript
   * import {
   *  Series,
   *  Int16Series,
   *  Int16
   * } from '@rapidsai/cudf';
   *
   * // Int16Series [1, 2, 3]
   * const a = Series.new(new Int16Array([1, 2, 3]));
   * const b = Series.new(new Int16Buffer([1, 2, 3]));
   * ```
   */
  static new(input: Int16Array|Int16Buffer): Series<Int16>;

  /**
   * Create a new cudf.Int32Series
   *
   * @example
   * ```typescript
   * import {
   *  Series,
   *  Int32Series,
   *  Int32
   * } from '@rapidsai/cudf';
   *
   * // Int32Series [1, 2, 3]
   * const a = Series.new(new Int32Array([1, 2, 3]));
   * const b = Series.new(new Int32Buffer([1, 2, 3]));
   * ```
   */
  static new(input: Int32Array|Int32Buffer): Series<Int32>;

  /**
   * Create a new cudf.Uint8Series
   *
   * @example
   * ```typescript
   * import {
   *  Series,
   *  Uint8Series,
   *  Uint8
   * } from '@rapidsai/cudf';
   *
   * // Uint8Series [1, 2, 3]
   * const a = Series.new(new Uint8Array([1, 2, 3]));
   * const b = Series.new(new Uint8Buffer([1, 2, 3]));
   * const c = Series.new(new Uint8ClampedArray([1, 2, 3]));
   * const d = Series.new(new Uint8ClampedBuffer([1, 2, 3]));
   * ```
   */
  static new(input: Uint8Array|Uint8Buffer|Uint8ClampedArray|Uint8ClampedBuffer): Series<Uint8>;

  /**
   * Create a new cudf.Uint16Series
   *
   * @example
   * ```typescript
   * import {
   *  Series,
   *  Uint16Series,
   *  Uint16
   * } from '@rapidsai/cudf';
   *
   * // Uint16Series [1, 2, 3]
   * const a = Series.new(new Uint16Array([1, 2, 3]));
   * const b = Series.new(new Uint16Buffer([1, 2, 3]));
   * ```
   */
  static new(input: Uint16Array|Uint16Buffer): Series<Uint16>;

  /**
   * Create a new cudf.Uint32Series
   *
   * @example
   * ```typescript
   * import {
   *  Series,
   *  Uint32Series,
   *  Uint32
   * } from '@rapidsai/cudf';
   *
   * // Uint32Series [1, 2, 3]
   * const a = Series.new(new Uint32Array([1, 2, 3]));
   * const b = Series.new(new Uint32Buffer([1, 2, 3]));
   * ```
   */
  static new(input: Uint32Array|Uint32Buffer): Series<Uint32>;

  /**
   * Create a new cudf.Uint64Series
   *
   * @example
   * ```typescript
   * import {
   *  Series,
   *  Uint64Series,
   *  Uint64
   * } from '@rapidsai/cudf';
   *
   * // Uint64Series [1n, 2n, 3n]
   * const a = Series.new(new BigUint64Array([1n, 2n, 3n]));
   * const b = Series.new(new Uint64Buffer([1n, 2n, 3n]));
   * ```
   */
  static new(input: BigUint64Array|Uint64Buffer): Series<Uint64>;

  /**
   * Create a new cudf.Float32Series
   *
   * @example
   * ```typescript
   * import {
   *  Series,
   *  Float32Series,
   *  Float32
   * } from '@rapidsai/cudf';
   *
   * // Float32Series [1, 2, 3]
   * const a = Series.new(new Float32Array([1, 2, 3]));
   * const b = Series.new(new Float32Buffer([1, 2, 3]));
   * ```
   */
  static new(input: Float32Array|Float32Buffer): Series<Float32>;

  /**
   * Create a new cudf.StringSeries
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * // StringSeries ["foo", "bar", "test", null]
   * const a = Series.new(["foo", "bar", "test", null]);
   * ```
   */
  static new(input: (string|null|undefined)[]): Series<Utf8String>;
  /**
   * Create a new cudf.Float64Series
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * // Float64Series [1, 2, 3, null, 4]
   * const a = Series.new([1, 2, 3, undefined, 4]);
   * ```
   */
  static new(input: (number|null|undefined)[]|Float64Array|Float64Buffer): Series<Float64>;
  /**
   * Create a new cudf.Int64Series
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * // Int64Series [1n, 2n, 3n, null, 4n]
   * const a = Series.new([1n, 2n, 3n, undefined, 4n]);
   * ```
   */
  static new(input: (bigint|null|undefined)[]|BigInt64Array|Int64Buffer): Series<Int64>;

  /**
   * Create a new cudf.Bool8Series
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * // Bool8Series [true, false, null, false]
   * const a = Series.new([true, false, undefined, false]);
   * ```
   */
  static new(input: (boolean|null|undefined)[]): Series<Bool8>;
  /**
   * Create a new cudf.TimestampMillisecondSeries
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * // TimestampMillisecondSeries [2021-05-13T00:00:00.000Z, null, 2021-05-13T00:00:00.000Z,
   * null] const a = Series.new([new Date(), undefined, new Date(), undefined]);
   * ```
   */
  static new(input: (Date|null|undefined)[]): Series<TimestampMillisecond>;
  /**
   * Create a new cudf.ListSeries that contain cudf.StringSeries elements.
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * // ListSeries [["foo", "bar"], ["test", null]]
   * const a = Series.new([["foo", "bar"], ["test",null]]);
   * a.getValue(0) // StringSeries ["foo", "bar"]
   * a.getValue(1) // StringSeries ["test", null]
   * ```
   */
  static new(input: (string|null|undefined)[][]): Series<List<Utf8String>>;
  /**
   * Create a new cudf.ListSeries that contain cudf.Float64Series elements.
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * // ListSeries [[1, 2], [3, null, 4]]
   * const a = Series.new([[1, 2], [3, undefined, 4]]);
   * a.getValue(0) // Float64Series [1, 2]
   * a.getValue(1) // Float64Series [3, null, 4]
   * ```
   */
  static new(input: (number|null|undefined)[][]): Series<List<Float64>>;
  /**
   * Create a new cudf.ListSeries that contain cudf.Int64Series elements.
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * // ListSeries [[1n, 2n], [3n, null, 4n]]
   * const a = Series.new([[1n, 2n], [3n, undefined, 4n]]);
   * a.getValue(0) // Int64Series [1n, 2n]
   * a.getValue(1) // Int64Series [3n, null, 4n]
   * ```
   */
  static new(input: (bigint|null|undefined)[][]): Series<List<Int64>>;
  /**
   * Create a new cudf.ListSeries that contain cudf.Bool8Series elements.
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * // ListSeries [[true, false], [null, false]]
   * const a = Series.new([[true, false], [undefined, false]]);
   * a.getValue(0) // Bool8Series [true, false]
   * a.getValue(1) // Bool8Series [null, false]
   * ```
   */
  static new(input: (boolean|null|undefined)[][]): Series<List<Bool8>>;
  /**
   * Create a new cudf.ListSeries that contain cudf.TimestampMillisecondSeries elements.
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * // ListSeries [[2021-05-13T00:00:00.000Z, null], [null, 2021-05-13T00:00:00.000Z]]
   * const a = Series.new([[new Date(), undefined], [undefined, new Date()]]);
   * a.getValue(0) // TimestampMillisecondSeries [2021-05-13T00:00:00.000Z, null]
   * a.getValue(1) // TimestampMillisecondSeries [null, 2021-05-13T00:00:00.000Z]
   * ```
   */
  static new(input: (Date|null|undefined)[][]): Series<List<TimestampMillisecond>>;

  static new<T extends readonly unknown[]>(input: T):
    Series<ArrowToCUDFType<JavaScriptArrayDataType<T>>>;

  static new<T extends DataType>(input: AbstractSeries<T>|Column<T>|SeriesProps<T>|arrow.Vector<T>|
                                 (string|null|undefined)[]|(number|null|undefined)[]|
                                 (bigint|null|undefined)[]|(boolean|null|undefined)[]|
                                 (Date|null|undefined)[]|(string|null|undefined)[][]|
                                 (number|null|undefined)[][]|(bigint|null|undefined)[][]|
                                 (boolean|null|undefined)[][]|(Date|null|undefined)[][]): Series<T>;

  static new<T extends DataType>(input: any) {
    return columnToSeries(asColumn<T>(input)) as any as Series<T>;
  }

  /**
   * Constructs a Series from a text file path.
   *
   * @note If delimiter is omitted, the default is ''.
   *
   * @param filepath Path of the input file.
   * @param delimiter Optional delimiter.
   *
   * @returns StringSeries from the file, split by delimiter.
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * const infile = Series.readText('./inputAsciiFile.txt')
   * ```
   */
  public static readText(filepath: string, delimiter: string): Series<Utf8String> {
    return Series.new(Column.readText(filepath, delimiter ?? ''));
  }

  /**
   * Constructs a Series with a sequence of values.
   *
   * @note If init is omitted, the default is 0.
   * @note If step is omitted, the default is 1.
   * @note If type is omitted, the default is Int32.
   *
   * @param opts Options for creating the sequence
   * @returns Series with the sequence
   *
   * @example
   * ```typescript
   * import {Series, Int64, Float32} from '@rapidsai/cudf';
   *
   * Series.sequence({size: 5}).toArray() // Int32Array[0, 1, 2, 3, 4]
   * Series.sequence({size: 5, init: 5}).toArray() // Int32Array[5, 6, 7, 8, 9]
   * Series
   *   .sequence({ size: 5, init: 0, type: new Int64 })
   *   .toArray() // BigInt64Array[0n, 1n, 2n, 3n, 4n]
   * Series
   *   .sequence({ size: 5, step: 2, init: 1, type: new Float32 })
   *   .toArray() // Float32Array[1, 3, 5, 7, 9]
   * ```
   */
  static sequence<U extends Numeric = Int32>(opts: {
    size: number;
    type?: U;  //
    init?: U['scalarType'];
    step?: U['scalarType'];
    memoryResource?: MemoryResource;
  }): Series<U> {
    const type = opts.type ?? new Int32;
    const init = new Scalar({type, value: <any>opts.init ?? 0}) as Scalar<U>;
    const step = new Scalar({type, value: <any>opts.step ?? 1}) as Scalar<U>;
    return Series.new(Column.sequence<U>(opts.size, init, step, opts.memoryResource));
  }

  /** @ignore */
  declare public _col: Column<T>;

  protected constructor(col: Column<T>) { DISPOSER.add(this._col = col); }

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
   * Casts the values to a new dtype (similar to `static_cast` in C++).
   *
   * @param dataType The new dtype.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns Series of same size as the current Series containing result of the `cast` operation.
   * @example
   * ```typescript
   * import {Series, Bool8, Int32} from '@rapidsai/cudf';
   *
   * const a = Series.new({type:new Int32, data: [1,0,1,0]});
   *
   * a.cast(new Bool8); // Bool8Series [true, false, true, false];
   * ```
   */
  cast<R extends DataType>(dataType: R, memoryResource?: MemoryResource): Series<R> {
    return new CastVisitor(this, memoryResource).visit(dataType);
  }

  // clang-format off
  /* eslint-disable @typescript-eslint/no-unused-vars */
  _castAsBool8(_memoryResource?: MemoryResource): Series<Bool8> { throw new Error(`cast from ${arrow.Type[this.type.typeId]} to Bool8 unimplemented`); }
  _castAsInt8(_memoryResource?: MemoryResource): Series<Int8> { throw new Error(`cast from ${arrow.Type[this.type.typeId]} to Int8 unimplemented`); }
  _castAsInt16(_memoryResource?: MemoryResource): Series<Int16> { throw new Error(`cast from ${arrow.Type[this.type.typeId]} to Int16 unimplemented`); }
  _castAsInt32(_memoryResource?: MemoryResource): Series<Int32> { throw new Error(`cast from ${arrow.Type[this.type.typeId]} to Int32 unimplemented`); }
  _castAsInt64(_memoryResource?: MemoryResource): Series<Int64> { throw new Error(`cast from ${arrow.Type[this.type.typeId]} to Int64 unimplemented`); }
  _castAsUint8(_memoryResource?: MemoryResource): Series<Uint8> { throw new Error(`cast from ${arrow.Type[this.type.typeId]} to Uint8 unimplemented`); }
  _castAsUint16(_memoryResource?: MemoryResource): Series<Uint16> { throw new Error(`cast from ${arrow.Type[this.type.typeId]} to Uint16 unimplemented`); }
  _castAsUint32(_memoryResource?: MemoryResource): Series<Uint32> { throw new Error(`cast from ${arrow.Type[this.type.typeId]} to Uint32 unimplemented`); }
  _castAsUint64(_memoryResource?: MemoryResource): Series<Uint64> { throw new Error(`cast from ${arrow.Type[this.type.typeId]} to Uint64 unimplemented`); }
  _castAsFloat32(_memoryResource?: MemoryResource): Series<Float32> { throw new Error(`cast from ${arrow.Type[this.type.typeId]} to Float32 unimplemented`); }
  _castAsFloat64(_memoryResource?: MemoryResource): Series<Float64> { throw new Error(`cast from ${arrow.Type[this.type.typeId]} to Float64 unimplemented`); }
  _castAsString(_memoryResource?: MemoryResource): Series<Utf8String> { throw new Error(`cast from ${arrow.Type[this.type.typeId]} to String unimplemented`); }
  _castAsTimeStampDay(_memoryResource?: MemoryResource): Series<TimestampDay> { throw new Error(`cast from ${arrow.Type[this.type.typeId]} to TimeStampDay unimplemented`); }
  _castAsTimeStampSecond(_memoryResource?: MemoryResource): Series<TimestampSecond> { throw new Error(`cast from ${arrow.Type[this.type.typeId]} to TimeStampSecond unimplemented`); }
  _castAsTimeStampMillisecond(_memoryResource?: MemoryResource): Series<TimestampMillisecond> { throw new Error(`cast from ${arrow.Type[this.type.typeId]} to TimeStampMillisecond unimplemented`); }
  _castAsTimeStampMicrosecond(_memoryResource?: MemoryResource): Series<TimestampMicrosecond> { throw new Error(`cast from ${arrow.Type[this.type.typeId]} to TimeStampMicrosecond unimplemented`); }
  _castAsTimeStampNanosecond(_memoryResource?: MemoryResource): Series<TimestampNanosecond> { throw new Error(`cast from ${arrow.Type[this.type.typeId]} to TimeStampNanosecond unimplemented`); }
  /* eslint-enable @typescript-eslint/no-unused-vars */
  // clang-format on

  _castAsCategorical<R extends Categorical>(type: R, memoryResource?: MemoryResource): Series<R> {
    const categories = scope(() => {
      const uniq = scope(() => {
        return new DataFrame({value: this, order: Series.sequence({size: this.length})})
          .groupBy({by: 'value'})
          .nth(0);
      });
      return uniq.sortValues({order: {ascending: true}}, memoryResource).get('value');
    }, [this]);

    const codes = this.encodeLabels(categories, undefined, undefined, memoryResource);

    return Series.new<R>({
      type,
      length: codes.length,
      nullMask: this.mask,
      children: [codes, categories.cast(type.dictionary)]
    });
  }

  /**
   * Concat a Series to the end of the caller, returning a new Series of a common dtype.
   *
   * @param other The Series to concat to the end of the caller.
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * Series.new([1, 2, 3]).concat(Series.new([4, 5, 6])) // [1, 2, 3, 4, 5, 6]
   * ```
   */
  concat<R extends Series<DataType>>(other: R, memoryResource?: MemoryResource):
    Series<CommonType<T, R['type']>> {
    type U     = typeof type;
    const type = findCommonType(this.type, other.type);
    const lhs  = <Column<U>>(compareTypes(type, this.type) ? this._col : this.cast(type)._col);
    const rhs  = <Column<U>>(compareTypes(type, other.type) ? other._col : other.cast(type)._col);
    return Series.new(lhs.concat(rhs, memoryResource));
  }

  /**
   * Return the number of non-null elements in the Series.
   *
   * @returns The number of non-null elements
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * Series.new([1, 2, 3]).countNonNulls(); // 3
   * Series.new([1, null, 3]).countNonNulls(); // 2
   * ```
   */
  countNonNulls(): number { return this._col.length - this._col.nullCount; }

  /**
   * @summary Explicitly free the device memory associated with this Series.
   */
  dispose() { this._col.dispose(); }

  /**
   * Encode the Series values into integer labels.
   *
   *
   * @param categories The optional Series of values to encode into integers. Defaults to the
   *   unique elements in this Series.
   * @param type The optional integer DataType to use for the returned Series. Defaults to
   *   Uint32.
   * @param nullSentinel The optional value used to indicate missing category. Defaults to -1.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's
   *   device memory.
   * @returns A sequence of encoded integer labels with values between `0` and `n-1`
   *   categories, and `nullSentinel` for any null values
   */
  encodeLabels<R extends Integral = Uint32>(categories: Series<T>         = this.unique(true),
                                            type: R                       = new Uint32 as R,
                                            nullSentinel: R['scalarType'] = -1,
                                            memoryResource?: MemoryResource): Series<R> {
    return scope(() => {
      try {
        // If there is a failure casting to the current dtype, catch the exception and return
        // encoded labels with all values set to `nullSentinel`, since this means the Column
        // cannot contain any of the encoded categories.
        if (!compareTypes(this.type, categories.type)) { categories = categories.cast(this.type); }
      } catch {
        return Series.sequence(
          {type, init: nullSentinel, step: 0, memoryResource, size: this.length});
      }

      // 1. Join this Series' values with the `categories` Series to determine the index
      // positions (i.e. `codes`) of the values to keep.
      const codes = scope(() => {
        const lhs = new DataFrame(
          {value: this, order: Series.sequence({type: new Uint32, size: this.length})});
        const rhs = new DataFrame(
          {value: categories, codes: Series.sequence({type, size: categories.length})});
        return lhs.join({on: ['value'], how: 'left', nullEquality: true, other: rhs})
          .drop(['value'])
          // 2. Sort the codes by the original value's position in this Series.
          .sortValues({order: {ascending: true}})
          .get('codes');
      });

      // 3. Replace missing codes with `nullSentinel`.
      return codes.replaceNulls(nullSentinel, memoryResource) as Series<R>;
    }, [this, categories]);
  }

  /**
   * Fills a range of elements in a column out-of-place with a scalar value.
   *
   * @param begin The starting index of the fill range (inclusive).
   * @param end The index of the last element in the fill range (exclusive), default
   *   this.length
   *   .
   * @param value The scalar value to fill.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's
   *   device memory.
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * // Float64Series
   * Series.new([1, 2, 3]).fill(0) // [0, 0, 0]
   * // StringSeries
   * Series.new(["foo", "bar", "test"]).fill("rplc", 0, 1) // ["rplc", "bar", "test"]
   * // Bool8Series
   * Series.new([true, true, true]).fill(false, 1) // [true, false, false]
   * ```
   */
  fill(value: T['scalarType'], begin = 0, end = this.length, memoryResource?: MemoryResource):
    Series<T> {
    return this.__construct(
      this._col.fill(new Scalar({type: this.type, value}), begin, end, memoryResource));
  }

  /**
   * Fills a range of elements in-place in a column with a scalar value.
   *
   * @param begin The starting index of the fill range (inclusive)
   * @param end The index of the last element in the fill range (exclusive)
   * @param value The scalar value to fill
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * // Float64Series
   * Series.new([1, 2, 3]).fillInPlace(0) // [0, 0, 0]
   * // StringSeries
   * Series.new(["foo", "bar", "test"]).fillInPlace("rplc", 0, 1) // ["rplc", "bar", "test"]
   * // Bool8Series
   * Series.new([true, true, true]).fillInPlace(false, 1) // [true, false, false]
   * ```
   */
  fillInPlace(value: T['scalarType'], begin = 0, end = this.length) {
    this._col.fillInPlace(new Scalar({type: this.type, value}), begin, end);
    return this;
  }

  /**
   * Replace null values with a scalar value.
   *
   * @param value The scalar value to use in place of nulls.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's
   *   device memory.
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * // Float64Series
   * Series.new([1, null, 3]).replaceNulls(-1) // [1, -1, 3]
   * // StringSeries
   * Series.new(["foo", "bar", null]).replaceNulls("rplc") // ["foo", "bar", "rplc"]
   * // Bool8Series
   * Series.new([null, true, true]).replaceNulls(false) // [true, true, true]
   * ```
   */
  replaceNulls(value: T['scalarType']|any, memoryResource?: MemoryResource): Series<T>;

  /**
   * Replace null values with the corresponding elements from another Series.
   *
   * @param value The Series to use in place of nulls.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's
   *   device memory.
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const replace = Series.new([10, 10, 10]);
   * const replaceBool = Series.new([false, false, false]);
   *
   * // Float64Series
   * Series.new([1, null, 3]).replaceNulls(replace) // [1, 10, 3]
   * // StringSeries
   * Series.new(["foo", "bar", null]).replaceNulls(replace) // ["foo", "bar", "10"]
   * // Bool8Series
   * Series.new([null, true, true]).replaceNulls(replaceBool) // [false, true, true]
   * ```
   */
  replaceNulls(value: Series<T>, memoryResource?: MemoryResource): Series<T>;

  replaceNulls(value: any, memoryResource?: MemoryResource): Series<T> {
    if (value instanceof Series) {
      return this.__construct(this._col.replaceNulls(value._col, memoryResource));
    } else {
      return this.__construct(
        this._col.replaceNulls(new Scalar({type: this.type, value}), memoryResource));
    }
  }

  /**
   * Replace null values with the non-null value following the null value in the same series.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Column's
   *   device memory.
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * // Float64Series
   * Series.new([1, null, 3]).replaceNullsFollowing() // [1, 3, 3]
   * // StringSeries
   * Series.new(["foo", "bar", null]).replaceNullsFollowing() // ["foo", "bar", null]
   * Series.new(["foo", null, "bar"]).replaceNullsFollowing() // ["foo", "bar", "bar"]
   * // Bool8Series
   * Series.new([null, true, true]).replaceNullsFollowing() // [true, true, true]
   * ```
   */
  replaceNullsFollowing(memoryResource?: MemoryResource): Series<T> {
    return this.__construct(this._col.replaceNulls(true, memoryResource));
  }

  /**
   * Replace null values with the non-null value preceding the null value in the same series.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Column's
   *   device memory.
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * // Float64Series
   * Series.new([1, null, 3]).replaceNullsPreceding() // [1, 1, 3]
   * // StringSeries
   * Series.new([null, "foo", "bar"]).replaceNullsPreceding() // [null, "foo", "bar"]
   * Series.new(["foo", null, "bar"]).replaceNullsPreceding() // ["foo", "foo", "bar"]
   * // Bool8Series
   * Series.new([true, null, false]).replaceNullsPreceding() // [true, true, false]
   * ```
   */
  replaceNullsPreceding(memoryResource?: MemoryResource): Series<T> {
    return this.__construct(this._col.replaceNulls(false, memoryResource));
  }

  /**
   * Returns a new series with reversed elements.
   *
   * @param memoryResource An optional MemoryResource used to allocate the result's device memory.
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * // Float64Series
   * Series.new([1, 2, 3]).reverse() // [3, 2, 1]
   * // StringSeries
   * Series.new(["foo", "bar"]).reverse() // ["bar", "foo"]
   * // Bool8Series
   * Series.new([false, true]).reverse() // [true, false]
   * ```
   */
  reverse(memoryResource?: MemoryResource): Series<T> {
    return this.gather(
      Series.sequence({size: this.length, step: -1, init: this.length - 1}), false, memoryResource);
  }

  /**
   * @summary Return sub-selection from a Series using the specified integral indices.
   *
   * @description Gathers the rows of the source columns according to `selection`, such that row "i"
   * in the resulting Series's columns will contain row `selection[i]` from the source columns. The
   * number of rows in the result series will be equal to the number of elements in selection. A
   * negative value i in the selection is interpreted as i+n, where `n` is the number of rows in
   * the source series.
   *
   * For dictionary columns, the keys column component is copied and not trimmed if the gather
   * results in abandoned key elements.
   *
   * @param indices A Series of 8/16/32-bit signed or unsigned integer indices to gather.
   * @param nullify_out_of_bounds If `true`, coerce rows that corresponds to out-of-bounds indices
   *   in the selection to null. If `false`, skips all bounds checking for selection values. Pass
   *   false if you are certain that the selection contains only valid indices for better
   *   performance. If `false` and there are out-of-bounds indices in the selection, the behavior
   *   is undefined. Defaults to `false`.
   * @param memoryResource An optional MemoryResource used to allocate the result's device memory.
   *
   * @example
   * ```typescript
   * import {Series, Int32} from '@rapidsai/cudf';
   *
   * const a = Series.new([1,2,3]);
   * const b = Series.new(["foo", "bar", "test"]);
   * const c = Series.new([true, false, true]);
   * const selection = Series.new({type: new Int32, data: [0,2]});
   *
   * a.gather(selection) // Float64Series [1,3]
   * b.gather(selection) // StringSeries ["foo", "test"]
   * c.gather(selection) // Bool8Series [true, true]
   * ```
   */
  gather(indices: Series<IndexType>|number[],
         nullify_out_of_bounds = false,
         memoryResource?: MemoryResource): Series<T> {
    const map = Array.isArray(indices) ? Series.new(indices).cast(new Uint32) : indices;
    return this.__construct(this._col.gather(map._col, nullify_out_of_bounds, memoryResource));
  }

  /**
   * Return a copy of this Series.
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * const a = Series.new(["foo", "bar", "test"]);
   *
   * a.copy() // StringSeries ["foo", "bar", "test"]
   * ```
   */
  copy(memoryResource?: MemoryResource): Series<T> {
    return this.__construct(this._col.copy(memoryResource));
  }

  /**
   * Returns the n largest element(s).
   *
   * @param n The number of values to retrieve.
   * @param keep Determines whether to keep the first or last of any duplicate values.
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * const a = Series.new([4, 6, 8, 10, 12, 1, 2]);
   * const b = Series.new(["foo", "bar", "test"]);
   *
   * a.nLargest(); // [12, 10, 8, 6, 4]
   * b.nLargest(1); // ["test"]
   * a.nLargest(-1); // []
   * ```
   */
  nLargest(n = 5, keep: keyof typeof DuplicateKeepOption = 'first'): Series<T> {
    return _nLargestOrSmallest(this as Series, true, n, keep);
  }

  /**
   * Returns the n smallest element(s).
   *
   * @param n The number of values to retrieve.
   * @param keep Determines whether to keep the first or last of any duplicate values.
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * const a = Series.new([4, 6, 8, 10, 12, 1, 2]);
   * const b = Series.new(["foo", "bar", "test"]);
   *
   * a.nSmallest(); // [1, 2, 4, 6, 8]
   * b.nSmallest(1); // ["bar"]
   * a.nSmallest(-1); // []
   * ```
   */
  nSmallest(n = 5, keep: keyof typeof DuplicateKeepOption = 'first'): Series<T> {
    return _nLargestOrSmallest(this as Series, false, n, keep);
  }

  /**
   * Returns the first n rows.
   *
   * @param n The number of rows to return.
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * const a = Series.new([4, 6, 8, 10, 12, 1, 2]);
   * const b = Series.new(["foo", "bar", "test"]);
   *
   * a.head(); // [4, 6, 8, 10, 12]
   * b.head(1); // ["foo"]
   * a.head(-1); // throws index out of bounds error
   * ```
   */
  head(n = 5): Series<T> {
    if (n < 0) { throw new Error('Index provided is out of bounds'); }
    const selection = Series.sequence(
      {type: new Int32, size: n < this._col.length ? n : this._col.length, init: 0});
    return this.__construct(this._col.gather(selection._col, false));
  }

  /**
   * Returns the last n rows.
   *
   * @param n The number of rows to return.
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * const a = Series.new([4, 6, 8, 10, 12, 1, 2]);
   * const b = Series.new(["foo", "bar", "test"]);
   *
   * a.tail(); // [8, 10, 12, 1, 2]
   * b.tail(1); // ["test"]
   * a.tail(-1); // throws index out of bounds error
   * ```
   */
  tail(n = 5): Series<T> {
    if (n < 0) { throw new Error('Index provided is out of bounds'); }
    const length = n < this._col.length ? n : this._col.length;
    const selection =
      Series.sequence({type: new Int32, size: length, init: this._col.length - length});
    return this.__construct(this._col.gather(selection._col, false));
  }

  /**
   * Scatters single value into this Series according to provided indices.
   *
   * @param value A column of values to be scattered in to this Series
   * @param indices A column of integral indices that indicate the rows in the this Series to be
   *   replaced by `value`.
   * @param check_bounds Optionally perform bounds checking on the indices and throw an error if
   *   any of its values are out of bounds (default: false).
   * @param memoryResource An optional MemoryResource used to allocate the result's device memory.
   *
   * @example
   * ```typescript
   * import {Series, Int32} from '@rapidsai/cudf';
   * const a = Series.new({type: new Int32, data: [0, 1, 2, 3, 4]});
   * const indices = Series.new({type: new Int32, data: [2, 4]});
   * const indices_out_of_bounds = Series.new({type: new Int32, data: [5, 6]});
   *
   * a.scatter(-1, indices); // returns [0, 1, -1, 3, -1];
   * a.scatter(-1, indices_out_of_bounds, true) // throws index out of bounds error
   *
   * ```
   */
  scatter(value: T['scalarType'],
          indices: Series<IndexType>|number[],
          check_bounds?: boolean,
          memoryResource?: MemoryResource): Series<T>;
  /**
   * Scatters a column of values into this Series according to provided indices.
   *
   * @param value A value to be scattered in to this Series
   * @param indices A column of integral indices that indicate the rows in the this Series to be
   *   replaced by `value`.
   * @param check_bounds Optionally perform bounds checking on the indices and throw an error if
   *   any of its values are out of bounds (default: false).
   * @param memoryResource An optional MemoryResource used to allocate the result's device memory.
   *
   * @example
   * ```typescript
   * import {Series, Int32} from '@rapidsai/cudf';
   * const a = Series.new({type: new Int32, data: [0, 1, 2, 3, 4]});
   * const b = Series.new({type: new Int32, data: [200, 400]});
   * const indices = Series.new({type: new Int32, data: [2, 4]});
   * const indices_out_of_bounds = Series.new({type: new Int32, data: [5, 6]});
   *
   * a.scatter(b, indices); // returns [0, 1, 200, 3, 400];
   * a.scatter(b, indices_out_of_bounds, true) // throws index out of bounds error
   * ```
   */
  scatter(values: Series<T>,
          indices: Series<IndexType>|number[],
          check_bounds?: boolean,
          memoryResource?: MemoryResource): Series<T>;

  scatter(source: Series<T>|T['scalarType'],
          indices: Series<IndexType>|number[],
          check_bounds = false,
          memoryResource?: MemoryResource): Series<T> {
    const dst = new Table({columns: [this._col]});
    const map = Array.isArray(indices) ? Series.new(indices).cast(new Uint32) : indices;
    if (source instanceof Series) {
      const src = new Table({columns: [source.cast(this.type)._col]});
      return this.__construct(
        dst.scatterTable(src, map._col, check_bounds, memoryResource).getColumnByIndex(0));
    }
    const src = [new Scalar({type: this.type, value: source})];
    return this.__construct(
      dst.scatterScalar(src, map._col, check_bounds, memoryResource).getColumnByIndex(0));
  }

  /**
   * Return a sub-selection of this Series using the specified boolean mask.
   *
   * @param mask A Series of boolean values for whose corresponding element in this Series
   *   will be selected or ignored.
   * @param memoryResource An optional MemoryResource used to allocate the result's device memory.
   *
   * @example
   * ```typescript
   * import {Series} from "@rapidsai/cudf";
   * const mask = Series.new([true, false, true]);
   *
   * // Float64Series
   * Series.new([1, 2, 3]).filter(mask) // [1, 3]
   * // StringSeries
   * Series.new(["foo", "bar", "test"]).filter(mask) // ["foo", "test"]
   * // Bool8Series
   * Series.new([false, true, true]).filter(mask) // [false, true]
   * ```
   */
  filter(mask: Series<Bool8>, memoryResource?: MemoryResource): Series<T> {
    return this.__construct(this._col.applyBooleanMask(mask._col, memoryResource));
  }

  /**
   * set values at the specified indices
   *
   * @param indices the indices in this Series to set values for
   * @param values the values to set at Series of indices
   *
   * @example
   * ```typescript
   * import {Series, Int32} from '@rapidsai/cudf';
   * const a = Series.new({type: new Int32, data: [0, 1, 2, 3, 4]});
   * const values = Series.new({type: new Int32, data: [200, 400]});
   * const indices = Series.new({type: new Int32, data: [2, 4]});
   *
   * a.setValues(indices, values); // inplace update [0, 1, 200, 3, 400];
   * a.setValues(indices, -1); // inplace update [0, 1, -1, 3, -1];
   * ```
   */
  setValues(indices: Series<Int32>|number[], values: Series<T>|T['scalarType']): void {
    this._col = this.scatter(values, indices)._col as Column<T>;
  }

  /**
   * Copy the underlying device memory to host, and return an Iterator of the values.
   */
  [Symbol.iterator](): IterableIterator<T['TValue']|null> {
    return this.toArrow()[Symbol.iterator]();
  }

  /**
   * Copy the underlying device memory to host and return an Array (or TypedArray) of the values.
   * @returns
   */
  toArray() { return this.toArrow().toArray(); }

  /**
   * Return a string with a tabular representation of the Series, pretty-printed according to the
   * options given.
   *
   * @param options
   */
  toString(options: DisplayOptions&{name?: string} = {}) {
    return new DataFrame({[options.name ?? '0']: this}).toString(options);
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
  setNullMask(mask: MemoryData|ArrayLike<number>|ArrayLike<bigint>, nullCount?: number) {
    this._col.setNullMask(mask, nullCount);
  }

  /**
   * Copy a Series to an Arrow vector in host memory
   */
  toArrow() {
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    return new DataFrame({0: this._col}).toArrow().getChildAt<T>(0)!;
  }

  /**
   * Generate an ordering that sorts the Series in a specified way.
   *
   * @param ascending whether to sort ascending (true) or descending (false)
   * @param null_order whether nulls should sort before or after other values
   * @param memoryResource An optional MemoryResource used to allocate the result's device memory.
   *
   * @returns Series containting the permutation indices for the desired sort order
   *
   * @example
   * ```typescript
   * import {Series, NullOrder} from '@rapidsai/cudf';
   *
   * // Float64Series
   * Series.new([50, 40, 30, 20, 10, 0]).orderBy(false) // [0, 1, 2, 3, 4, 5]
   * Series.new([50, 40, 30, 20, 10, 0]).orderBy(true) // [5, 4, 3, 2, 1, 0]
   *
   * // StringSeries
   * Series.new(["a", "b", "c", "d", "e"]).orderBy(false) // [4, 3, 2, 1, 0]
   * Series.new(["a", "b", "c", "d", "e"]).orderBy(true) // [0, 1, 2, 3, 4]
   *
   * // Bool8Series
   * Series.new([true, false, true, true, false]).orderBy(true) // [1, 4, 0, 2, 3]
   * Series.new([true, false, true, true, false]).orderBy(false) // [0, 2, 3, 1, 4]
   *
   * // NullOrder usage
   * Series.new([50, 40, 30, 20, 10, null]).orderBy(false, 'before') // [0, 1, 2, 3, 4, 5]
   * Series.new([50, 40, 30, 20, 10, null]).orderBy(false, 'after') // [5, 0, 1, 2, 3, 4]
   * ```
   */
  orderBy(ascending                          = true,
          null_order: keyof typeof NullOrder = 'after',
          memoryResource?: MemoryResource) {
    return Series.new(new Table({
                        columns: [this._col]
                      }).orderBy([ascending], [null_order === 'before'], memoryResource));
  }

  /**
   * Generate a new Series that is sorted in a specified way.
   *
   * @param ascending whether to sort ascending (true) or descending (false)
   *   Default: true
   * @param null_order whether nulls should sort before or after other values
   *   Default: before
   * @param memoryResource An optional MemoryResource used to allocate the result's device memory.
   *
   * @returns Sorted values
   *
   * @example
   * ```typescript
   * import {Series, NullOrder} from '@rapidsai/cudf';
   *
   * // Float64Series
   * Series.new([50, 40, 30, 20, 10, 0]).sortValues(false) // [50, 40, 30, 20, 10, 0]
   * Series.new([50, 40, 30, 20, 10, 0]).sortValues(true) // [0, 10, 20, 30, 40, 50]
   *
   * // StringSeries
   * Series.new(["a", "b", "c", "d", "e"]).sortValues(false) // ["e", "d", "c", "b", "a"]
   * Series.new(["a", "b", "c", "d", "e"]).sortValues(true) // ["a", "b", "c", "d", "e"]
   *
   * // Bool8Series
   * Series.new([true, false, true, true, false]).sortValues(true) // [false, false, true,
   * true, true] Series.new([true, false, true, true, false]).sortValues(false) // [true,
   * true, true, false, false]
   *
   * // NullOrder usage
   * Series.new([50, 40, 30, 20, 10, null]).sortValues(false, 'before') // [50, 40, 30, 20,
   * 10, null]
   *
   * Series.new([50, 40, 30, 20, 10, null]).sortValues(false, 'after') // [null, 50, 40, 30,
   * 20, 10]
   * ```
   */
  sortValues(ascending                          = true,
             null_order: keyof typeof NullOrder = 'after',
             memoryResource?: MemoryResource): Series<T> {
    return this.gather(this.orderBy(ascending, null_order), false, memoryResource);
  }

  /**
   * Creates a Series of `BOOL8` elements where `true` indicates the value is null and `false`
   * indicates the value is valid.
   *
   * @param memoryResource Memory resource used to allocate the result Column's device memory.
   * @returns A non-nullable Series of `BOOL8` elements with `true` representing `null`
   *   values.
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * // Float64Series
   * Series.new([1, null, 3]).isNull() // [false, true, false]
   * // StringSeries
   * Series.new(["foo", "bar", null]).isNull() // [false, false, true]
   * // Bool8Series
   * Series.new([true, true, null]).isNull() // [false, false, true]
   * ```
   */
  isNull(memoryResource?: MemoryResource) { return Series.new(this._col.isNull(memoryResource)); }

  /**
   * Creates a Series of `BOOL8` elements where `true` indicates the value is valid and
   * `false` indicates the value is null.
   *
   * @param memoryResource Memory resource used to allocate the result Column's device memory.
   * @returns A non-nullable Series of `BOOL8` elements with `false` representing `null`
   *   values.
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * // Float64Series
   * Series.new([1, null, 3]).isNotNull() // [true, false, true]
   * // StringSeries
   * Series.new(["foo", "bar", null]).isNotNull() // [true, true, false]
   * // Bool8Series
   * Series.new([true, true, null]).isNotNull() // [true, true, false]
   * ```
   */
  isNotNull(memoryResource?: MemoryResource) {
    return Series.new(this._col.isValid(memoryResource));
  }

  /**
   * drop Null values from the series
   * @param memoryResource Memory resource used to allocate the result Column's device memory.
   * @returns series without Null values
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * // Float64Series
   * Series.new([1, undefined, 3]).dropNulls() // [1, 3]
   * Series.new([1, null, 3]).dropNulls() // [1, 3]
   * Series.new([1, , 3]).dropNulls() // [1, 3]
   *
   * // StringSeries
   * Series.new(["foo", "bar", undefined]).dropNulls() // ["foo", "bar"]
   * Series.new(["foo", "bar", null]).dropNulls() // ["foo", "bar"]
   * Series.new(["foo", "bar", ,]).dropNulls() // ["foo", "bar"]
   *
   * // Bool8Series
   * Series.new([true, true, undefined]).dropNulls() // [true, true]
   * Series.new([true, true, null]).dropNulls() // [true, true]
   * Series.new([true, true, ,]).dropNulls() // [true, true]
   * ```
   */
  dropNulls(memoryResource?: MemoryResource): Series<T> {
    return this.__construct(this._col.dropNulls(memoryResource));
  }

  /**
   * @summary Hook for specialized Series to override when constructing from a C++ Column.
   */
  protected __construct(col: Column<T>): Series<T> {
    return Series.new(Object.assign(col, {type: transferFields(this.type, col.type)}));
  }

  /**
   * Returns an object with keys "value" and "count" whose respective values are new Series
   * containing the unique values in the original series and the number of times they occur
   * in the original series.
   * @returns object with keys "value" and "count"
   */
  valueCounts(): {count: Int32Series, value: Series<T>} {
    const df = new DataFrame({count: this, value: this}).groupBy({by: 'value'}).count();
    return {
      count: df.get('count'),
      value: df.get('value'),
    };
  }

  /**
   * Returns a new Series with only the unique values that were found in the original
   *
   * @param nullsEqual Determines whether nulls are handled as equal values.
   * @param memoryResource Memory resource used to allocate the result Column's device memory.
   * @returns series without duplicate values
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * // Float64Series
   * Series.new([null, null, 1, 2, 3, 3]).unique(true) // [null, 1, 2, 3]
   * Series.new([null, null, 1, 2, 3, 3]).unique(false) // [null, null, 1, 2, 3]
   * ```
   */
  unique(nullsEqual = true, memoryResource?: MemoryResource) {
    return this.dropDuplicates(true, nullsEqual, true, memoryResource);
  }

  /**
   * Returns a new Series with duplicate values from the original removed
   *
   * @param keep Determines whether or not to keep the duplicate items.
   * @param nullsEqual Determines whether nulls are handled as equal values.
   * @param nullsFirst Determines whether null values are inserted before or after non-null
   *   values.
   * @param memoryResource Memory resource used to allocate the result Column's device memory.
   * @returns series without duplicate values
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * // Float64Series
   * Series.new([4, null, 1, 2, null, 3, 4]).dropDuplicates(
   *   true,
   *   true,
   *   true
   * ) // [null, 1, 2, 3, 4]
   *
   * Series.new([4, null, 1, 2, null, 3, 4]).dropDuplicates(
   *   false,
   *   true,
   *   true
   * ) // [1, 2, 3]
   * ```
   */
  dropDuplicates(keep       = true,
                 nullsEqual = true,
                 nullsFirst = true,
                 memoryResource?: MemoryResource) {
    return new DataFrame({0: this})
      .dropDuplicates(keep ? 'first' : 'none', nullsEqual, nullsFirst, ['0'], memoryResource)
      .get('0');
  }
}

// eslint-disable-next-line @typescript-eslint/no-redeclare
export const Series = AbstractSeries;

import {Bool8Series} from './series/bool';
import {CategoricalSeries} from './series/categorical';
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
import {
  TimestampDaySeries,
  TimestampMicrosecondSeries,
  TimestampMillisecondSeries,
  TimestampNanosecondSeries,
  TimestampSecondSeries
} from './series/timestamp';

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

function inferType(value: any[]): DataType {
  if (value.length === 0) { return new Float64; }
  let nullsCount    = 0;
  let arraysCount   = 0;
  let objectsCount  = 0;
  let numbersCount  = 0;
  let stringsCount  = 0;
  let bigintsCount  = 0;
  let booleansCount = 0;
  let datesCount    = 0;
  value.forEach((val) => {
    if (val == null) { return ++nullsCount; }
    switch (typeof val) {
      case 'bigint': return ++bigintsCount;
      case 'boolean': return ++booleansCount;
      case 'number': return ++numbersCount;
      case 'string': return ++stringsCount;
      case 'object':
        if (Array.isArray(val)) {
          return ++arraysCount;
        } else if (Object.prototype.toString.call(val) === '[object Date]') {
          return ++datesCount;
        } else {
          return ++objectsCount;
        }
    }
    throw new TypeError(
      'Unable to infer Series type from input values, explicit type declaration expected');
  });

  if (numbersCount + nullsCount === value.length) {
    return new Float64;
  } else if (stringsCount + nullsCount === value.length) {
    return new Utf8String;
  } else if (bigintsCount + nullsCount === value.length) {
    return new Int64;
  } else if (booleansCount + nullsCount === value.length) {
    return new Bool8;
  } else if (datesCount + nullsCount === value.length) {
    return new TimestampMillisecond;
  } else if (arraysCount + nullsCount === value.length) {
    const childType = inferType(value[value.findIndex((ary) => ary != null)]);
    if (value.every((ary) => ary == null || compareTypes(childType, inferType(ary)))) {
      return new List(new arrow.Field('', childType));
    }
  } else if (objectsCount + nullsCount === value.length) {
    const fields = new Map<string, arrow.Field>();
    value.forEach((val) => {
      Object.keys(val).forEach((key) => {
        if (!fields.has(key)) {
          // use the type inferred for the first instance of a found key
          fields.set(key, new arrow.Field(key, inferType([val[key]]), true));
        }
      });
    }, {});
    return new Struct<any>([...fields.values()]);
  }

  throw new TypeError(
    'Unable to infer Series type from input values, explicit type declaration expected');
}

const arrayToDtype = (value: any[]|MemoryView|ArrayBufferView) => {
  switch (value.constructor) {
    case Int8Array: return new Int8;
    case Int8Buffer: return new Int8;
    case Int16Array: return new Int16;
    case Int16Buffer: return new Int16;
    case Int32Array: return new Int32;
    case Int32Buffer: return new Int32;
    case BigInt64Array: return new Int64;
    case Int64Buffer: return new Int64;
    case Uint8Array: return new Uint8;
    case Uint8Buffer: return new Uint8;
    case Uint8ClampedArray: return new Uint8;
    case Uint8ClampedBuffer: return new Uint8;
    case Uint16Array: return new Uint16;
    case Uint16Buffer: return new Uint16;
    case Uint32Array: return new Uint32;
    case Uint32Buffer: return new Uint32;
    case BigUint64Array: return new Uint64;
    case Uint64Buffer: return new Uint64;
    case Float32Array: return new Float32;
    case Float32Buffer: return new Float32;
    case Float64Array: return new Float64;
    case Float64Buffer: return new Float64;
    default: return inferType(value as any[]);
  }
};

function asColumn(value: Int8Array|Int8Buffer): Column<Int8>;
function asColumn(value: Int16Array|Int16Buffer): Column<Int16>;
function asColumn(value: Int32Array|Int32Buffer): Column<Int32>;
function asColumn(value: BigInt64Array|Int64Buffer): Column<Int64>;
function asColumn(value: Uint8Array|Uint8Buffer): Column<Uint8>;
function asColumn(value: Uint8ClampedArray|Uint8ClampedBuffer): Column<Uint8>;
function asColumn(value: Uint16Array|Uint16Buffer): Column<Uint16>;
function asColumn(value: Uint32Array|Uint32Buffer): Column<Uint32>;
function asColumn(value: BigUint64Array|Uint64Buffer): Column<Uint64>;
function asColumn(value: Float32Array|Float32Buffer): Column<Float32>;
function asColumn(value: Float64Array|Float64Buffer): Column<Float64>;

function asColumn<T extends DataType>(value: AbstractSeries<T>|SeriesProps<T>  //
                                      |Column<T>|arrow.Vector<T>               //
                                      |(string | null | undefined)[]           //
                                      |(number | null | undefined)[]           //
                                      |(bigint | null | undefined)[]           //
                                      |(boolean | null | undefined)[]          //
                                      |(Date | null | undefined)[]             //
                                      |(string | null | undefined)[][]         //
                                      |(number | null | undefined)[][]         //
                                      |(bigint | null | undefined)[][]         //
                                      |(boolean | null | undefined)[][]        //
                                      |(Date | null | undefined)[][]): Column<T>;

function asColumn<T extends DataType>(value: any) {
  if (!value) { value = []; }

  // Return early if it's already a Series
  if (value instanceof AbstractSeries) { return value._col; }

  // Return early if it's already a Column
  if (value instanceof Column) { return value as Column<T>; }

  // If Array/Vector/TypedArray/MemoryView, wrap in a ColumnProps
  if (value instanceof arrow.Vector) {
    value = {data: value};
  } else if (Array.isArray(value) ||       //
             ArrayBuffer.isView(value) ||  //
             value instanceof MemoryView) {
    value = {data: value, type: arrayToDtype(value)};
  }

  let {data, offset} = value;

  if (typeof data !== 'object') {
    data = undefined;
  } else {
    // Use C++ Arrow-to-cuDF conversion if `data` is an Arrow Vector
    if (data instanceof arrow.Vector) {
      // Slice `offset` before converting it to a Column
      return fromArrow<T>(typeof offset !== 'number' ? data : data.slice(offset));
    }

    // If `data` is an Array, convert it to a Vector and use C++ Arrow-to-cuDF conversion
    if (Array.isArray(data)) {
      return fromArrow<T>(arrow.vectorFromArray(
        typeof offset !== 'number' ? data : data.slice(offset), value.type ?? inferType(data)));
    }

    // If `data.buffer` is a ArrayBuffer, copy it to a DeviceBuffer
    if (ArrayBuffer.isView(value) || (data.buffer instanceof ArrayBuffer)) {
      if (typeof data.length === 'number') { value.length = data.length; }
      data   = new DeviceBuffer(typeof offset !== 'number' ? data : data.subarray(offset));
      offset = 0;
    }
    // If `data.buffer` is a DeviceBuffer, propagate its `byteOffset` to ColumnProps
    else if (data.buffer instanceof DeviceBuffer) {
      if (typeof data.length === 'number') { value.length = data.length; }
      offset =
        (typeof offset !== 'number' ? 0 : offset) + (data.byteOffset / data.BYTES_PER_ELEMENT);
    }
  }

  return new Column<T>({...value, data, offset, children: value.children?.map(asColumn)});
}

const columnToSeries = (() => {
  interface ColumnToSeriesVisitor extends arrow.Visitor {
    visit<T extends DataType>(column: Column<T>): Series<T>;
    visitMany<T extends DataType>(columns: Column<T>[]): Series<T>[];
    getVisitFn<T extends DataType>(column: Column<T>): (column: Column<T>) => Series<T>;
  }
  /* eslint-disable @typescript-eslint/no-unused-vars */
  class ColumnToSeriesVisitor extends arrow.Visitor {
    public visit<T extends DataType>(col: Column<T>): Series<T> {
      for (let i = -1, n = col.numChildren; ++i < n;) {
        // Visit each child to ensure concrete dtypes
        this.visit(col.getChild(i));
      }
      return super.visit(col);
    }
    public getVisitFn<T extends DataType>(column: Column<T>): (column: Column<T>) => Series<T> {
      let {type} = column;
      if (!(type instanceof arrow.DataType)) {
        type = {...(<any>type), __proto__: arrow.DataType.prototype};
      }
      return super.getVisitFn(type);
    }
    public visitBool<T extends Bool8>(col: Column<T>) {
      return new (<any>Bool8Series)(Object.assign(col, {type: new Bool8}));
    }
    public visitInt8<T extends Int8>(col: Column<T>) {
      return new (<any>Int8Series)(Object.assign(col, {type: new Int8}));
    }
    public visitInt16<T extends Int16>(col: Column<T>) {
      return new (<any>Int16Series)(Object.assign(col, {type: new Int16}));
    }
    public visitInt32<T extends Int32>(col: Column<T>) {
      return new (<any>Int32Series)(Object.assign(col, {type: new Int32}));
    }
    public visitInt64<T extends Int64>(col: Column<T>) {
      return new (<any>Int64Series)(Object.assign(col, {type: new Int64}));
    }
    public visitUint8<T extends Uint8>(col: Column<T>) {
      return new (<any>Uint8Series)(Object.assign(col, {type: new Uint8}));
    }
    public visitUint16<T extends Uint16>(col: Column<T>) {
      return new (<any>Uint16Series)(Object.assign(col, {type: new Uint16}));
    }
    public visitUint32<T extends Uint32>(col: Column<T>) {
      return new (<any>Uint32Series)(Object.assign(col, {type: new Uint32}));
    }
    public visitUint64<T extends Uint64>(col: Column<T>) {
      return new (<any>Uint64Series)(Object.assign(col, {type: new Uint64}));
    }
    public visitFloat32<T extends Float32>(col: Column<T>) {
      return new (<any>Float32Series)(Object.assign(col, {type: new Float32}));
    }
    public visitFloat64<T extends Float64>(col: Column<T>) {
      return new (<any>Float64Series)(Object.assign(col, {type: new Float64}));
    }
    public visitUtf8<T extends Utf8String>(col: Column<T>) {
      return new (<any>StringSeries)(Object.assign(col, {type: new Utf8String}));
    }
    public visitDateDay<T extends TimestampDay>(col: Column<T>) {
      return new (<any>TimestampDaySeries)(Object.assign(col, {type: new TimestampDay}));
    }
    public visitDateMillisecond<T extends TimestampMillisecond>(col: Column<T>) {
      return new (<any>TimestampMillisecondSeries)(
        Object.assign(col, {type: new TimestampMillisecond}));
    }
    public visitTimestampSecond<T extends TimestampSecond>(col: Column<T>) {
      return new (<any>TimestampSecondSeries)(Object.assign(col, {type: new TimestampSecond}));
    }
    public visitTimestampMillisecond<T extends TimestampMillisecond>(col: Column<T>) {
      return new (<any>TimestampMillisecondSeries)(
        Object.assign(col, {type: new TimestampMillisecond}));
    }
    public visitTimestampMicrosecond<T extends TimestampMicrosecond>(col: Column<T>) {
      return new (<any>TimestampMicrosecondSeries)(
        Object.assign(col, {type: new TimestampMicrosecond}));
    }
    public visitTimestampNanosecond<T extends TimestampNanosecond>(col: Column<T>) {
      return new (<any>TimestampNanosecondSeries)(
        Object.assign(col, {type: new TimestampNanosecond}));
    }
    public visitList<T extends List>(col: Column<T>) {
      const {type, nullable} = col.getChild(1);  // elements
      const {name, metadata} = col.type.children[0];
      const childField       = arrow.Field.new({name, type, nullable, metadata});
      return new (<any>ListSeries)(Object.assign(col, {type: new List(childField)}));
    }
    public visitStruct<T extends Struct>(col: Column<T>) {
      const childFields = col.type.children.map(({name, metadata}, i) => {
        const {type, nullable} = col.getChild(i);
        return arrow.Field.new({name, type, nullable, metadata});
      });
      return new (<any>StructSeries)(Object.assign(col, {type: new Struct(childFields)}));
    }
    public visitDictionary<T extends Categorical>(col: Column<T>) {
      const {id, isOrdered}    = col.type;
      const {type: dictionary} = col.getChild(1);  // categories
      return new (<any>CategoricalSeries)(
        Object.assign(col, {type: new Categorical(dictionary, id, isOrdered)}));
    }
    // clang-format off
    // public visitBinary               <T extends Binary>(col: Column<T>) { return new (BinarySeries as any)(col); }
    // public visitFixedSizeBinary      <T extends FixedSizeBinary>(col: Column<T>) { return new (FixedSizeBinarySeries as any)(col); }
    // public visitTimeSecond           <T extends TimeSecond>(col: Column<T>) { return new (TimeSecondSeries as any)(col); }
    // public visitTimeMillisecond      <T extends TimeMillisecond>(col: Column<T>) { return new (TimeMillisecondSeries as any)(col); }
    // public visitTimeMicrosecond      <T extends TimeMicrosecond>(col: Column<T>) { return new (TimeMicrosecondSeries as any)(col); }
    // public visitTimeNanosecond       <T extends TimeNanosecond>(col: Column<T>) { return new (TimeNanosecondSeries as any)(col); }
    // public visitDecimal              <T extends Decimal>(col: Column<T>) { return new (DecimalSeries as any)(col); }
    // public visitDenseUnion           <T extends DenseUnion>(col: Column<T>) { return new (DenseUnionSeries as any)(col); }
    // public visitSparseUnion          <T extends SparseUnion>(col: Column<T>) { return new (SparseUnionSeries as any)(col); }
    // public visitIntervalDayTime      <T extends IntervalDayTime>(col: Column<T>) { return new (IntervalDayTimeSeries as any)(col); }
    // public visitIntervalYearMonth    <T extends IntervalYearMonth>(col: Column<T>) { return new (IntervalYearMonthSeries as any)(col); }
    // public visitFixedSizeList        <T extends FixedSizeList>(col: Column<T>) { return new (FixedSizeListSeries as any)(col); }
    // public visitMap                  <T extends Map>(col: Column<T>) { return new (MapSeries as any)(col); }
    // clang-format on
  }
  /* eslint-enable @typescript-eslint/no-unused-vars */
  const visitor = new ColumnToSeriesVisitor();
  return function columnToSeries<T extends DataType>(column: Column<T>) {
    return visitor.visit(column);
  };
})();

function _nLargestOrSmallest<T extends DataType>(
  series: Series<T>, ascending: boolean, n: number, keep: keyof typeof DuplicateKeepOption):
  Series {
  if (keep == 'first') {
    return series.sortValues(!ascending).head(n < 0 ? 0 : n);
  } else if (keep == 'last') {
    return n <= 0 ? Series.new({type: series.type, data: new Array(0)})
                  : series.sortValues(ascending).tail(n).reverse();
  } else {
    throw new TypeError('keep must be either "first" or "last"');
  }
}

function transferFields<T extends DataType>(lhs: T, rhs: T) {
  const {children: lhsFields} = lhs;
  const {children: rhsFields} = rhs;
  if (lhsFields && rhsFields && lhsFields.length && rhsFields.length) {
    lhsFields.forEach(({name, type, nullable, metadata}, i) => {
      rhsFields[i] = arrow.Field.new({
        name,
        nullable,
        metadata,
        type: transferFields(type, rhsFields[i].type),
      });
    });
  }
  return rhs;
}
