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

import {MemoryData, MemoryView, Uint8Buffer} from '@rapidsai/cuda';
import {DeviceBuffer, MemoryResource} from '@rapidsai/rmm';
import * as arrow from 'apache-arrow';
import {compareTypes} from 'apache-arrow/visitor/typecomparator';
import {Readable} from 'stream';

import {Column} from './column';
import {ColumnAccessor} from './column_accessor';
import {concat as concatDataFrames} from './dataframe/concat';
import {Join, JoinResult} from './dataframe/join';
import {DataFrameFormatter, DisplayOptions} from './dataframe/print';
import {GroupByMultiple, GroupByMultipleProps, GroupBySingle, GroupBySingleProps} from './groupby';
import {Scalar} from './scalar';
import {Series} from './series';
import {Table, ToArrowMetadata} from './table';
import {CSVTypeMap, ReadCSVOptions, WriteCSVOptions} from './types/csv';
import {
  Bool8,
  DataType,
  FloatingPoint,
  FloatTypes,
  IndexType,
  Int32,
  Int64,
  Integral,
  IntegralTypes,
  Numeric,
  NumericTypes,
} from './types/dtypes';
import {DuplicateKeepOption, NullOrder} from './types/enums';
import {ColumnsMap, CommonType, TypeMap} from './types/mappings';
import {ReadORCOptions, WriteORCOptions} from './types/orc';
import {ReadParquetOptions, WriteParquetOptions} from './types/parquet';

export type SeriesMap<T extends TypeMap = any> = {
  [P in keyof T]: {readonly type: T[P]}
};

export type OrderSpec = {
  ascending?: boolean,
  null_order?: keyof typeof NullOrder
};

type JoinType = 'inner'|'outer'|'left'|'right'|'leftsemi'|'leftanti';

type JoinProps<
  Rhs extends TypeMap,
  TOn extends string,
  How extends JoinType = 'inner',
  LSuffix extends string = '',
  RSuffix extends string = '',
> = {
  other: DataFrame<Rhs>;
  on: TOn[];
  how?: How;
  lsuffix?: LSuffix;
  rsuffix?: RSuffix;
  nullEquality?: boolean;
  memoryResource?: MemoryResource;
};

type CombinedGroupByProps<T extends TypeMap, R extends keyof T, IndexKey extends string> =
  GroupBySingleProps<T, R>|Partial<GroupByMultipleProps<T, R, IndexKey>>;

function _seriesToColumns<T extends TypeMap>(data: ColumnsMap<T>|SeriesMap<T>) {
  const columns = {} as any;
  for (const [name, col] of Object.entries(data)) {
    if (col instanceof Column) {
      columns[name] = col;
    } else if (col instanceof Series) {
      columns[name] = col._col;
    } else {
      columns[name] = Series.new(col)._col;
    }
  }
  return <ColumnsMap<T>>columns;
}

function _throwIfNonNumeric(type: DataType, operationName: string) {
  if (!NumericTypes.some((t) => compareTypes(t, type))) {
    throw new TypeError(`dtype ${type.toString()} cannot perform the operation: ${operationName}`);
  }
}

/**
 * A GPU Dataframe object.
 */
export class DataFrame<T extends TypeMap = any> {
  /**
   * Read CSV files from disk and create a cudf.DataFrame
   *
   * @example
   * ```typescript
   * import {DataFrame, Series, Int16, Bool, Float32, Utf8String}  from '@rapidsai/cudf';
   * const df = DataFrame.readCSV({
   *  header: 0,
   *  sourceType: 'files',
   *  sources: ['test.csv'],
   *  dataTypes: {
   *    a: new Int16,
   *    b: new Bool,
   *    c: new Float32,
   *    d: new Utf8String
   *  }
   * })
   * ```
   */
  public static readCSV<T extends CSVTypeMap = any>(options: ReadCSVOptions<T>) {
    const {names, table} = Table.readCSV(options);
    return new DataFrame(new ColumnAccessor(
      names.reduce((map, name, i) => ({...map, [name]: table.getColumnByIndex(i)}),
                   {} as ColumnsMap<{[P in keyof T]: T[P]}>)));
  }

  /**
   * Read Apache ORC files from disk and create a cudf.DataFrame
   *
   * @example
   * ```typescript
   * import {DataFrame}  from '@rapidsai/cudf';
   * const df = DataFrame.readORC({
   *  sourceType: 'files',
   *  sources: ['test.orc'],
   * })
   * ```
   */
  public static readORC(options: ReadORCOptions) {
    const {names, table} = Table.readORC(options);
    return new DataFrame(new ColumnAccessor(
      names.reduce((map, name, i) => ({...map, [name]: table.getColumnByIndex(i)}), {})));
  }

  /**
   * Read Apache Parquet files from disk and create a cudf.DataFrame
   *
   * @example
   * ```typescript
   * import {DataFrame}  from '@rapidsai/cudf';
   * const df = DataFrame.readParquet({
   *  sourceType: 'files',
   *  sources: ['test.parquet'],
   * })
   * ```
   */
  public static readParquet(options: ReadParquetOptions) {
    const {names, table} = Table.readParquet(options);
    return new DataFrame(new ColumnAccessor(
      names.reduce((map, name, i) => ({...map, [name]: table.getColumnByIndex(i)}), {})));
  }

  /**
   * Adapts an Arrow Table in IPC format into a DataFrame.
   *
   * @param memory A buffer holding Arrow table
   * @return The Arrow data as a DataFrame
   */
  public static fromArrow<T extends TypeMap>(memory: DeviceBuffer|MemoryData): DataFrame<T> {
    if (memory instanceof ArrayBuffer || ArrayBuffer.isView(memory)) {
      memory = new Uint8Buffer(memory);
    }
    if (memory instanceof MemoryView) { memory = memory.buffer; }
    const {names, table} = Table.fromArrow(memory);
    return new DataFrame(new ColumnAccessor(names.reduce(
      (map, name, i) => ({...map, [name]: table.getColumnByIndex(i)}), {} as ColumnsMap<T>)));
  }

  declare private _accessor: ColumnAccessor<T>;

  /**
   * Create a new cudf.DataFrame
   *
   * @example
   * ```typescript
   * import {DataFrame, Series}  from '@rapidsai/cudf';
   * const df = new DataFrame({
   *  a: Series.new([1, 2]),
   *  b: Series.new([true, false]),
   *  c: Series.new(["foo", "bar"])
   * })
   *
   * ```
   */
  constructor(data?: SeriesMap<T>);
  constructor(data?: ColumnsMap<T>);
  constructor(data?: ColumnAccessor<T>);
  constructor(data: any = {}) {
    this._accessor =
      (data instanceof ColumnAccessor) ? data : new ColumnAccessor(_seriesToColumns(data));
  }

  /**
   * The number of rows in each column of this DataFrame
   *
   * @example
   * ```typescript
   * import {DataFrame, Series}  from '@rapidsai/cudf';
   * const df = new DataFrame({
   *  a: Series.new([1, 2]),
   *  b: Series.new([1, 2]),
   *  c: Series.new([1, 2])
   * })
   *
   * df.numRows // 2
   * ```
   */
  get numRows() { return this._accessor.columns.length > 0 ? this._accessor.columns[0].length : 0; }

  /**
   * The number of columns in this DataFrame
   *
   * @example
   * ```typescript
   * import {DataFrame, Series}  from '@rapidsai/cudf';
   * const df = new DataFrame({
   *  a: Series.new([1, 2]),
   *  b: Series.new([1, 2]),
   *  c: Series.new([1, 2])
   * })
   *
   * df.numColumns // 3
   * ```
   */
  get numColumns() { return this._accessor.length; }

  /**
   * The names of columns in this DataFrame
   *
   * @example
   * ```typescript
   * import {DataFrame, Series}  from '@rapidsai/cudf';
   * const df = new DataFrame({
   *  a: Series.new([1, 2]),
   *  b: Series.new([1, 2]),
   *  c: Series.new([1, 2])
   * })
   *
   * df.names // ['a', 'b', 'c']
   * ```
   */
  get names() { return this._accessor.names; }

  /**
   * A map of this DataFrame's Series names to their DataTypes
   *
   * @example
   * ```typescript
   * import {DataFrame, Series}  from '@rapidsai/cudf';
   * const df = new DataFrame({
   *  a: Series.new([1, 2]),
   *  b: Series.new(["foo", "bar"]),
   *  c: Series.new([[1, 2], [3]]),
   * })
   *
   * df.types
   * // {
   * //   a: [Object Float64],
   * //   b: [Object Utf8String],
   * //   c: [Object List]
   * // }
   * ```
   */
  get types() { return this._accessor.types; }

  /** @ignore */
  asTable() { return new Table({columns: this._accessor.columns}); }

  /**
   * Return a string with a tabular representation of the DataFrame, pretty-printed according to the
   * options given.
   *
   * @param options
   */
  toString(options: DisplayOptions = {}) { return new DataFrameFormatter(options, this).render(); }

  /**
   * Return a new DataFrame containing only specified columns.
   *
   * @param columns Names of columns keep.
   *
   * @example
   * ```typescript
   * import {DataFrame, Series}  from '@rapidsai/cudf';
   * const df = new DataFrame({
   *  a: Series.new([0, 1, 1, 2, 2, 2]),
   *  b: Series.new([0, 1, 2, 3, 4, 4]),
   *  c: Series.new([1, 2, 3, 4, 5, 6])
   * })
   *
   * df.select(['a', 'b']) // returns df with {a, b}
   * ```
   */
  select<R extends keyof T>(names: readonly R[]) {
    return new DataFrame(this._accessor.selectByColumnNames(names));
  }

  /**
   * Return a new DataFrame with new columns added.
   *
   * @param data mapping of names to new columns to add
   *
   * @example
   * ```typescript
   * import {DataFrame, Series} from '@rapidsai/cudf';
   *
   * const df = new DataFrame({a: [1, 2, 3]});
   *
   * df.assign({b: Series.new(["foo", "bar", "bar"])})
   * // returns df {a: [1, 2, 3], b: ["foo", "bar", "bar"]}
   * ```
   */
  assign<R extends TypeMap>(data: SeriesMap<R>): DataFrame<Omit<T, keyof R&string>&R>;

  /**
   * Return a new DataFrame with new columns added.
   *
   * @param data a GPU DataFrame object
   *
   * @example
   * ```typescript
   * import {DataFrame} from '@rapidsai/cudf';
   *
   * const df = new DataFrame({a: [1, 2, 3]});
   * const df1 = new DataFrame({b: ["foo", "bar", "bar"]});
   *
   * df.assign(df1) // returns df {a: [1, 2, 3], b: ["foo", "bar", "bar"]}
   * ```
   */
  assign<R extends TypeMap>(data: DataFrame<R>): DataFrame<Omit<T, keyof R&string>&R>;

  assign<R extends TypeMap>(data: SeriesMap<R>|DataFrame<R>) {
    const columns = (data instanceof DataFrame) ? data._accessor : _seriesToColumns(data);
    return new DataFrame(this._accessor.addColumns(columns));
  }

  /**
   * Return a new DataFrame with specified columns removed.
   *
   * @param names Names of the columns to drop.
   *
   * @example
   * ```typescript
   * import {DataFrame, Series, Int32, Float32}  from '@rapidsai/cudf';
   * const df = new DataFrame({
   *  a: Series.new({type: new Int32, data: [0, 1, 1, 2, 2, 2]}),
   *  b: Series.new({type: new Float32, data: [0, 1, 2, 3, 4, 4]})
   * });
   *
   * df.drop(['a']) // returns df {b: [0, 1, 2, 3, 4, 4]}
   * ```
   */
  drop<R extends keyof T>(names: readonly R[]) {
    return new DataFrame(this._accessor.dropColumns(names));
  }

  /**
   * Return a new DataFrame with specified columns renamed.
   *
   * @param nameMap Object mapping old to new Column names.
   *
   * @example
   * ```typescript
   * import {DataFrame, Series, Int32, Float32}  from '@rapidsai/cudf';
   * const df = new DataFrame({
   *  a: Series.new({type: new Int32, data: [0, 1, 1, 2, 2, 2]}),
   *  b: Series.new({type: new Float32, data: [0, 1, 2, 3, 4, 4]})
   * });
   *
   * df.rename({a: 'c'}) // returns df {b: [0, 1, 2, 3, 4, 4], c: [0, 1, 1, 2, 2, 2]}
   * ```
   */
  rename<U extends string|number, P extends {[K in keyof T]?: U}>(nameMap: P) {
    const names = Object.keys(nameMap) as (string & keyof P)[];
    return this.drop(names).assign(names.reduce(
      (xs, x) =>                                                                  //
      ({...xs, [`${nameMap[x]!}`]: this.get(x)}),                                 //
      {} as SeriesMap<{[K in keyof P as `${NonNullable<P[K]>}`]: T[string & K]}>  //
      ));
  }

  /**
   * Return whether the DataFrame has a Series.
   *
   * @param name Name of the Series to return.
   *
   * @example
   * ```typescript
   * import {DataFrame, Series, Int32, Float32}  from '@rapidsai/cudf';
   * const df = new DataFrame({
   *  a: Series.new({type: new Int32, data: [0, 1, 1, 2, 2, 2]}),
   *  b: Series.new({type: new Float32, data: [0, 1, 2, 3, 4, 4]})
   * });
   *
   * df.has('a') // true
   * df.has('c') // false
   * ```
   */
  has(name: string) { return this._accessor.has(name); }

  /**
   * Return a series by name.
   *
   * @param name Name of the Series to return.
   *
   * @example
   * ```typescript
   * import {DataFrame, Series, Int32, Float32}  from '@rapidsai/cudf';
   * const df = new DataFrame({
   *  a: Series.new({type: new Int32, data: [0, 1, 1, 2, 2, 2]}),
   *  b: Series.new({type: new Float32, data: [0, 1, 2, 3, 4, 4]})
   * });
   *
   * df.get('a') // Int32Series
   * df.get('b') // Float32Series
   * ```
   */
  get<P extends keyof T>(name: P): Series<T[P]> { return Series.new(this._accessor.get(name)); }

  /**
   * Casts each selected Series in this DataFrame to a new dtype (similar to `static_cast` in C++).
   *
   * @param dataTypes The map from column names to new dtypes.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns DataFrame of Series cast to the new dtype
   *
   * @example
   * ```typescript
   * import {DataFrame, Series, Int32, Float32}  from '@rapidsai/cudf';
   * const df = new DataFrame({
   *  a: Series.new({type: new Int32, data: [0, 1, 1, 2, 2, 2]}),
   *  b: Series.new({type: new Int32, data: [0, 1, 2, 3, 4, 4]})
   * });
   *
   * df.cast({a: new Float32}); // returns df with a as Float32Series and b as Int32Series
   * ```
   */
  cast<R extends {[P in keyof T]?: DataType}>(dataTypes: R, memoryResource?: MemoryResource) {
    const names = this._accessor.names;
    const types = !(dataTypes instanceof arrow.DataType)
                    ? dataTypes
                    : names.reduce((types, name) => ({...types, [name]: dataTypes}), {} as R);
    return new DataFrame(names.reduce(
      (columns, name) => ({
        ...columns,
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        [name]: name in types ? this.get(name).cast(types[name]!, memoryResource) : this.get(name)
      }),
      {} as SeriesMap<Omit<T, keyof R>&R>));
  }

  /**
   * Casts all the Series in this DataFrame to a new dtype (similar to `static_cast` in C++).
   *
   * @param dataType The new dtype.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns DataFrame of Series cast to the new dtype
   *make notebooks.run
   *  a: Series.new({type: new Int32, data: [0, 1, 1, 2, 2, 2]}),
   *  b: Series.new({type: new Int32, data: [0, 1, 2, 3, 4, 4]})
   * })
   *
   * df.castAll(new Float32); // returns df with a and b as Float32Series
   * ```
   */
  castAll<R extends DataType>(dataType: R, memoryResource?: MemoryResource) {
    return new DataFrame(this._accessor.names.reduce(
      (columns, name) => ({...columns, [name]: this.get(name).cast(dataType, memoryResource)}),
      {} as SeriesMap<{[P in keyof T]: R}>));
  }

  /**
   * Concat DataFrame(s) to the end of the caller, returning a new DataFrame.
   *
   * @param others The DataFrame(s) to concat to the end of the caller.
   *
   * @example
   * ```typescript
   * import {DataFrame, Series} from '@rapidsai/cudf';
   * const df = new DataFrame({
   *   a: Series.new([1, 2, 3, 4]),
   *   b: Series.new([1, 2, 3, 4]),
   * });
   *
   * const df2 = new DataFrame({
   *   a: Series.new([5, 6, 7, 8]),
   * });
   *
   * df.concat(df2);
   * // return {
   * //    a: [1, 2, 3, 4, 5, 6, 7, 8],
   * //    b: [1, 2, 3, 4, null, null, null, null],
   * // }
   * ```
   */
  concat<U extends DataFrame[]>(...others: U) { return concatDataFrames(this, ...others); }

  /**
   * @summary Explicitly free the device memory associated with this DataFrame.
   */
  dispose() {
    this.names.forEach((name) => this.get(name).dispose());
    this._accessor = new ColumnAccessor({} as ColumnsMap<T>);
  }

  /**
   * @summary Interleave columns of a DataFrame into a single Series.
   *
   * @param dataType The dtype of the result Series (required if the DataFrame has mixed dtypes).
   * @param memoryResource An optional MemoryResource used to allocate the result's device memory.
   *
   * @returns Series representing a packed row-major matrix of all the source DataFrame's Series.
   *
   * @example
   * ```typescript
   * import {DataFrame, Series }  from '@rapidsai/cudf';
   *
   * new DataFrame({
   *  a: Series.new([1, 2, 3]),
   *  b: Series.new([4, 5, 6]),
   * }).interleaveColumns()
   * // Float64Series [
   * //  1, 4, 2, 5, 3, 6
   * // ]
   *
   * new DataFrame({
   *  b: Series.new([ [0,  1,  2],  [3,  4,  5],  [6,  7,  8]]),
   *  c: Series.new([[10, 11, 12], [13, 14, 15], [16, 17, 18]]),
   * }).interleaveColumns()
   * // ListSeries [
   * //   [0,  1,  2],
   * //  [10, 11, 12],
   * //   [3,  4,  5],
   * //  [13, 14, 15],
   * //   [6,  7,  8],
   * //  [16, 17, 18],
   * // ]
   *
   */
  interleaveColumns<R extends T[keyof T] = T[keyof T]>(dataType?: R|null,
                                                       memoryResource?: MemoryResource) {
    return Series.new<R>(
      (dataType ? this.castAll(dataType) : this).asTable().interleaveColumns(memoryResource));
  }

  /**
   * Generate an ordering that sorts DataFrame columns in a specified way
   *
   * @param options mapping of column names to sort order specifications
   *
   * @returns Series containting the permutation indices for the desired sort order
   *
   * @example
   * ```typescript
   * import {DataFrame, Series, Int32, NullOrder}  from '@rapidsai/cudf';
   * const df = new DataFrame({a: Series.new([null, 4, 3, 2, 1, 0])});
   *
   * df.orderBy({a: {ascending: true, null_order: 'before'}});
   * // Int32Series [0, 5, 4, 3, 2, 1]
   *
   * df.orderBy({a: {ascending: true, null_order: 'after'}});
   * // Int32Series [5, 4, 3, 2, 1, 0]
   *
   * df.orderBy({a: {ascending: false, null_order: 'before'}});
   * // Int32Series [1, 2, 3, 4, 5, 0]
   *
   * df.orderBy({a: {ascending: false, null_order: 'after'}});
   * // Int32Series [0, 1, 2, 3, 4, 5]
   * ```
   */
  orderBy<R extends keyof T>(options: {[P in R]: OrderSpec}) {
    const column_orders = new Array<boolean>();
    const null_orders   = new Array<NullOrder>();
    const columns       = new Array<Column<T[keyof T]>>();
    const entries       = Object.entries(options) as [R, OrderSpec][];
    entries.forEach(([name, {ascending = true, null_order = 'after'}]) => {
      const child = this.get(name);
      if (child) {
        columns.push(child._col as Column<T[keyof T]>);
        column_orders.push(ascending);
        null_orders.push(NullOrder[null_order]);
      }
    });
    // Compute the sorted sorted_indices
    const sorted_indices = new Table({columns}).orderBy(column_orders, null_orders);
    return Series.new(sorted_indices);
  }

  /**
   * Generate a new DataFrame sorted in the specified way.
   *
   * @param ascending whether to sort ascending (true) or descending (false)
   *   Default: true
   * @param null_order whether nulls should sort before or after other values
   *   Default: after
   *
   * @returns A new DataFrame of sorted values
   *
   * @example
   * ```typescript
   * import {DataFrame, Series, Int32, NullOrder}  from '@rapidsai/cudf';
   * const df = new DataFrame({
   *   a: Series.new([null, 4, 3, 2, 1, 0]),
   *   b: Series.new([0, 1, 2, 3, 4, 5])
   * });
   *
   * df.sortValues({a: {ascending: true, null_order: 'after'}})
   * // {a: [0, 1, 2, 3, 4, null], b: [5, 4, 3, 2, 1, 0]}
   *
   * df.sortValues({a: {ascending: true, null_order: 'before'}})
   * // {a: [null, 0, 1, 2, 3, 4], b: [0, 5, 4, 3, 2, 1]}
   *
   * df.sortValues({a: {ascending: false, null_order: 'after'}})
   * // {a: [4, 3, 2, 1, 0, null], b: [1, 2, 3, 4, 5, 0]}
   *
   * df.sortValues({a: {ascending: false, null_order: 'before'}})
   * // {a: [null, 4, 3, 2, 1, 0], b: [0, 1, 2, 3, 4, 5]}
   * ```
   */
  sortValues<R extends keyof T>(options: {[P in R]: OrderSpec}) {
    return this.gather(this.orderBy(options));
  }

  /**
   * Return sub-selection from a DataFrame from the specified indices
   *
   * @param selection
   *
   * @example
   * ```typescript
   * import {DataFrame, Series, Int32}  from '@rapidsai/cudf';
   * const df = new DataFrame({
   *   a: Series.new({type: new Int32, data: [0, 1, 2, 3, 4, 5]}),
   *   b: Series.new([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
   * });
   *
   * const selection = Series.new({type: new Int32, data: [2,4,5]});
   *
   * df.gather(selection); // {a: [2, 4, 5], b: [2.0, 4.0, 5.0]}
   * ```
   */
  gather<R extends IndexType>(selection: Series<R>, nullify_out_of_bounds = false) {
    const temp       = new Table({columns: this._accessor.columns});
    const columns    = temp.gather(selection._col, nullify_out_of_bounds);
    const series_map = {} as SeriesMap<T>;
    this._accessor.names.forEach(
      (name, index) => { series_map[name] = Series.new(columns.getColumnByIndex(index)); });
    return new DataFrame(series_map);
  }

  /**
   * Returns the first n rows as a new DataFrame.
   *
   * @param n The number of rows to return.
   *
   * @example
   * ```typescript
   * import {DataFrame, Series, Int32} from '@rapidsai/cudf';
   *
   * const df = new DataFrame({
   *   a: Series.new({type: new Int32, data: [0, 1, 2, 3, 4, 5, 6]}),
   *   b: Series.new([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
   * });
   *
   * a.head();
   * // {a: [0, 1, 2, 3, 4], b: [0.0, 1.0, 2.0, 3.0, 4.0]}
   *
   * b.head(1);
   * // {a: [0], b: [0.0]}
   *
   * a.head(-1);
   * // throws index out of bounds error
   * ```
   */
  head(n = 5): DataFrame<T> {
    if (n < 0) { throw new Error('Index provided is out of bounds'); }
    const selection =
      Series.sequence({type: new Int32, size: n < this.numRows ? n : this.numRows, init: 0});
    return this.gather(selection);
  }

  /**
   * Returns the last n rows as a new DataFrame.
   *
   * @param n The number of rows to return.
   *
   * @example
   * ```typescript
   * import {DataFrame, Series, Int32} from '@rapidsai/cudf';
   *
   * const df = new DataFrame({
   *   a: Series.new({type: new Int32, data: [0, 1, 2, 3, 4, 5, 6]}),
   *   b: Series.new([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
   * });
   *
   * a.tail();
   * // {a: [2, 3, 4, 5, 6], b: [2.0, 3.0, 4.0, 5.0, 6.0]}
   *
   * b.tail(1);
   * // {a: [6], b: [6.0]}
   *
   * a.tail(-1);
   * // throws index out of bounds error
   * ```
   */
  tail(n = 5): DataFrame<T> {
    if (n < 0) { throw new Error('Index provided is out of bounds'); }
    const length    = n < this.numRows ? n : this.numRows;
    const selection = Series.sequence({type: new Int32, size: length, init: this.numRows - length});
    return this.gather(selection);
  }

  /**
   * Return a group-by on a single column.
   *
   * @param props configuration for the groupby
   *
   * @example
   * ```typescript
   * import {DataFrame, Series}  from '@rapidsai/cudf';
   * const df = new DataFrame({
   *  a: Series.new([0, 1, 1, 2, 2, 2]),
   *  b: Series.new([0, 1, 2, 3, 4, 4]),
   *  c: Series.new([1, 2, 3, 4, 5, 6])
   * })
   *
   * df.groupby({by: 'a'}).max() // { a: [2, 1, 0], b: [4, 2, 0], c: [6, 3, 1] }
   *
   * ```
   */
  groupBy<R extends keyof T>(props: GroupBySingleProps<T, R>): GroupBySingle<T, R>;

  /**
   * Return a group-by on a multiple columns.
   *
   * @param props configuration for the groupby
   *
   * @example
   * ```typescript
   * import {DataFrame, Series}  from '@rapidsai/cudf';
   * const df = new DataFrame({
   *  a: Series.new([0, 1, 1, 2, 2, 2]),
   *  b: Series.new([0, 1, 2, 3, 4, 4]),
   *  c: Series.new([1, 2, 3, 4, 5, 6])
   * })
   *
   * df.groupby({by: ['a', 'b']}).max()
   * // {
   * //   "a_b": [{"a": [2, 1, 1, 2, 0], "b": [4, 2, 1, 3, 0]}],
   * //   "c": [6, 3, 2, 4, 1]
   * // }
   *
   * ```
   */
  groupBy<R extends keyof T, IndexKey extends string>(props: GroupByMultipleProps<T, R, IndexKey>):
    GroupByMultiple<T, R, IndexKey>;

  groupBy<R extends keyof T, IndexKey extends string>(props: CombinedGroupByProps<T, R, IndexKey>) {
    if (!Array.isArray(props.by)) {
      return new GroupBySingle(this, props as GroupBySingleProps<T, R>);
    } else if ('index_key' in props) {
      return new GroupByMultiple(this, props as GroupByMultipleProps<T, R, IndexKey>);
    } else {
      return new GroupByMultiple(this, {
        ...props,
        index_key: props.by.join('_'),
      } as GroupByMultipleProps<T, R, any>);
    }
  }

  /**
   * Return sub-selection from a DataFrame from the specified boolean mask
   *
   * @param mask
   *
   * @example
   * ```typescript
   * import {DataFrame, Series, Bool8}  from '@rapidsai/cudf';
   * const df = new DataFrame({
   *  a: Series.new([0, 1, 2, 3, 4, 4]),
   *  b: Series.new([0, NaN, 2, 3, 4, 4])
   * })
   * const mask = Series.new({type: new Bool8, data: [0, 0, 1, 0, 1, 1]})
   *
   * df.filter(mask); // {a: [2, 4, 4], b: [2, 4, 4]}
   *
   * ```
   */
  filter(mask: Series<Bool8>) {
    const temp       = new Table({columns: this._accessor.columns});
    const columns    = temp.gather(mask._col, false);
    const series_map = {} as SeriesMap<T>;
    this._accessor.names.forEach(
      (name, index) => { series_map[name] = Series.new(columns.getColumnByIndex(index)); });
    return new DataFrame(series_map);
  }

  /**
   * Join columns with other DataFrame.
   *
   * @param props the configuration for the join
   * @returns the joined DataFrame
   */
  // clang-format off
  join<R extends TypeMap, TOn extends (string & keyof T & keyof R), LSuffix extends string = '', RSuffix extends string = ''>(
    props: JoinProps<R, TOn, 'inner'|'outer'|'left'|'right', LSuffix, RSuffix>
  ): DataFrame<{
    [P in keyof JoinResult<T, R, TOn, LSuffix, RSuffix>]:
      P extends TOn
        ? CommonType<T[P], R[P]>
        : JoinResult<T, R, TOn, LSuffix, RSuffix>[P]
  }>;
  // clang-format on

  /**
   * Join columns with other DataFrame.
   *
   * @param props the configuration for the join
   * @returns the joined DataFrame
   */
  // clang-format off
  join<R extends TypeMap, TOn extends (string & keyof T & keyof R)>(
    props: JoinProps<R, TOn, 'leftsemi'|'leftanti'>
  ): DataFrame<T>;
  // clang-format on

  // clang-format off
  join(props: any): any {
    // clang-format on
    const {how = 'inner', other, ...opts} = props;
    switch (how) {
      case 'left': return new Join({...opts, lhs: this, rhs: other}).left();
      case 'right': return new Join({...opts, lhs: this, rhs: other}).right();
      case 'inner': return new Join({...opts, lhs: this, rhs: other}).inner();
      case 'outer': return new Join({...opts, lhs: this, rhs: other}).outer();
      case 'leftsemi': return new Join({...opts, lhs: this, rhs: other}).leftSemi();
      case 'leftanti': return new Join({...opts, lhs: this, rhs: other}).leftAnti();
    }
  }

  /**
   * Serialize this DataFrame to CSV format.
   *
   * @param options Options controlling CSV writing behavior.
   *
   * @returns A node ReadableStream of the CSV data.
   */
  toCSV(options: WriteCSVOptions = {}) {
    const readable = new Readable({encoding: 'utf8'});
    new Table({columns: this._accessor.columns}).writeCSV({
      ...options,
      next(buf) { readable.push(buf); },
      complete() { readable.push(null); },
      columnNames: this.names as string[],
    });
    return readable as AsyncIterable<string>;
  }

  /**
   * Write a DataFrame to ORC format.
   *
   * @param filePath File path or root directory path.
   * @param options Options controlling ORC writing behavior.
   *
   */
  toORC(filePath: string, options: WriteORCOptions = {}) {
    this.asTable().writeORC(filePath, {...options, columnNames: this.names as string[]});
  }

  /**
   * Write a DataFrame to Parquet format.
   *
   * @param filePath File path or root directory path.
   * @param options Options controlling Parquet writing behavior.
   *
   */
  toParquet(filePath: string, options: WriteParquetOptions = {}) {
    this.asTable().writeParquet(filePath, {...options, columnNames: this.names as string[]});
  }

  /**
   * Copy a Series to an Arrow vector in host memory
   *
   * @example
   * ```typescript
   * import {DataFrame, Series} from "@rapidsai/cudf";
   *
   * const df = new DataFrame({a: Series.new([0,1,2]), b: Series.new(["one", "two", "three"])});
   *
   * const arrow_df = df.toArrow(); // Arrow table
   *
   * arrow_df.toArray();
   * // [
   * //    { "a": 0, "b": "one" },
   * //    { "a": 1, "b": "two" },
   * //    { "a": 2, "b": "three" }
   * //  ]
   * ```
   */
  toArrow() {
    const toArrowMetadata = (name: string|number, type?: DataType): ToArrowMetadata => {
      if (!type || !type.children || !type.children.length) { return [name]; }
      if (type instanceof arrow.List) {
        if (!type.children[0]) { return [name, [[0], [1]]]; }
        return [name, [[0], toArrowMetadata(type.children[0].name, type.children[0].type)]];
      }
      return [name, type.children.map((f) => toArrowMetadata(f.name, f.type))];
    };
    const names = this.names.map(
      (name, i) => toArrowMetadata(<string|number>name, this._accessor.columns[i].type));
    return arrow.Table.from<T>(this.asTable().toArrow(names));
  }

  /**
   * drop null rows
   * @ignore
   */
  protected _dropNullsRows(thresh = 1, subset = this.names) {
    const column_indices: number[] = [];
    const allNames                 = this.names;
    subset.forEach((col) => {
      if (allNames.includes(col)) {
        column_indices.push(allNames.indexOf(col));
      } else {
        throw new Error(`Unknown column name: ${col.toString()}`);
      }
    });

    const table_result = new Table({columns: this._accessor.columns});
    const result       = table_result.dropNulls(column_indices, thresh);
    return new DataFrame(
      allNames.reduce((map, name, i) => ({...map, [name]: Series.new(result.getColumnByIndex(i))}),
                      {} as SeriesMap<T>));
  }
  /**
   * drop rows with NaN values (float type only)
   * @ignore
   */
  protected _dropNaNsRows(thresh = 1, subset = this.names) {
    const column_indices: number[] = [];
    const allNames                 = this.names;
    subset.forEach((col) => {
      if (allNames.includes(col) && FloatTypes.some((t) => compareTypes(this.get(col).type, t))) {
        column_indices.push(allNames.indexOf(col));
      } else if (!allNames.includes(col)) {
        throw new Error(`Unknown column name: ${col.toString()}`);
      } else {
        // col exists but not of floating type
        thresh -= 1;
      }
    });
    const table_result = new Table({columns: this._accessor.columns});
    const result       = table_result.dropNans(column_indices, thresh);
    return new DataFrame(
      allNames.reduce((map, name, i) => ({...map, [name]: Series.new(result.getColumnByIndex(i))}),
                      {} as SeriesMap<T>));
  }
  /**
   * drop columns with nulls
   * @ignore
   */
  protected _dropNullsColumns(thresh = 1, subset?: Series) {
    const column_names: (keyof T)[] = [];
    const df                        = (subset !== undefined) ? this.gather(subset) : this;

    this.names.forEach(col => {
      const no_threshold_valid_count = (df.get(col).length - df.get(col).nullCount) < thresh;
      if (!no_threshold_valid_count) { column_names.push(col as string); }
    });

    return new DataFrame(column_names.reduce(
      (map, name) => ({...map, [name]: Series.new(this._accessor.get(name))}), {} as SeriesMap<T>));
  }
  /**
   * drop columns with NaN values(float type only)
   * @ignore
   */
  protected _dropNaNsColumns(thresh = 1, subset?: Series, memoryResource?: MemoryResource) {
    const column_names: (keyof T)[] = [];
    const df                        = (subset !== undefined) ? this.gather(subset) : this;

    this.names.forEach(col => {
      if (FloatTypes.some((t) => compareTypes(this.get(col).type, t))) {
        const nanCount =
          df.get(col)._col.nansToNulls(memoryResource).nullCount - this.get(col).nullCount;

        const no_threshold_valid_count = (df.get(col).length - nanCount) < thresh;
        if (!no_threshold_valid_count) { column_names.push(col); }
      } else {
        column_names.push(col);
      }
    });

    return new DataFrame(column_names.reduce(
      (map, name) => ({...map, [name]: Series.new(this._accessor.get(name))}), {} as SeriesMap<T>));
  }

  /**
   * Drops rows (or columns) containing nulls (*Note: only null values are dropped and not NaNs)
   *
   * @param axis Whether to drop rows (axis=0, default) or columns (axis=1) containing nulls
   * @param thresh drops every row (or column) containing less than thresh non-null values.
   *
   * thresh=1 (default) drops rows (or columns) containing all null values (non-null < thresh(1)).
   *
   * if axis = 0, thresh=df.numColumns: drops only rows containing at-least one null value
   * (non-null values in a row < thresh(df.numColumns)).
   *
   * if axis = 1, thresh=df.numRows: drops only columns containing at-least one null values
   * (non-null values in a column < thresh(df.numRows)).
   *
   * @param subset List of columns to consider when dropping rows (all columns are considered by
   *   default).
   * Alternatively, when dropping columns, subset is a Series<Integer> with indices to select rows
   * (all rows are considered by default).
   * @returns DataFrame<T> with dropped rows (or columns) containing nulls
   *
   * @example
   * ```typescript
   * import {DataFrame, Series}  from '@rapidsai/cudf';
   * const df = new DataFrame({
   *  a: Series.new([0, null, 2, null, 4, 4]),
   *  b: Series.new([0, null, 2, 3, null, 4]),
   *  c: Series.new([null, null, null, null, null, null])
   * });
   *
   * // delete rows with all nulls (default thresh=1)
   * df.dropNulls(0);
   * // return {
   * //   a: [0, 2, null, 4, 4], b: [0, 2, 3, null, 4],
   * //   c: [null, null, null, null, null]
   * // }
   *
   * // delete rows with atleast one null
   * df.dropNulls(0, df.numColumns);
   * // returns empty df, since each row contains atleast one null
   *
   * // delete columns with all nulls (default thresh=1)
   * df.dropNulls(1);
   * // returns {a: [0, null, 2, null, 4, 4], b: [0, null, 2, 3, null, 4]}
   *
   * // delete columns with atleast one null
   * df.dropNulls(1, df.numRows);
   * // returns empty df, since each column contains atleast one null
   *
   * ```
   */
  dropNulls<R extends IndexType>(axis = 0, thresh = 1, subset?: (string&keyof T)[]|Series<R>):
    DataFrame<T> {
    if (axis == 0) {
      if (subset instanceof Series) {
        throw new Error(
          'for axis=0, expected \'subset\' to be one of {list of column_names, undefined(all columns)}');
      }
      return this._dropNullsRows(thresh, subset);
    } else if (axis == 1) {
      if (subset instanceof Array) {
        throw new Error(
          'for axis=1, expected \'subset\' to be one of {Series<Integer> with indices to select rows, undefined(all rows)}');
      }
      return this._dropNullsColumns(thresh, subset);
    } else {
      throw new Error('invalid axis value, expected {0, 1} ');
    }
  }

  /**
   * Drops rows (or columns) containing NaN, provided the columns are of type float
   *
   * @param axis Whether to drop rows (axis=0, default) or columns (axis=1) containing NaN
   * @param thresh drops every row (or column) containing less than thresh non-NaN values.
   *
   * thresh=1 (default) drops rows (or columns) containing all NaN values (non-NaN < thresh(1)).
   *
   * if axis = 0, thresh=df.numColumns: drops only rows containing at-least one NaN value (non-NaN
   * values in a row < thresh(df.numColumns)).
   *
   * if axis = 1, thresh=df.numRows: drops only columns containing at-least one NaN values
   * (non-NaN values in a column < thresh(df.numRows)).
   *  @param subset List of float columns to consider when dropping rows (all float columns are
   *   considered by default).
   * Alternatively, when dropping columns, subset is a Series<Integer> with indices to select rows
   * (all rows are considered by default).
   *
   * @returns DataFrame<T> with dropped rows (or columns) containing NaN
   *
   * @example
   * ```typescript
   * import {DataFrame, Series}  from '@rapidsai/cudf';
   * const df = new DataFrame({
   *  a: Series.new([0, NaN, 2, NaN, 4, 4]),
   *  b: Series.new([0, NaN, 2, 3, NaN, 4]),
   *  c: Series.new([NaN, NaN, NaN, NaN, NaN, NaN])
   * });
   *
   * // delete rows with all NaNs (default thresh=1)
   * df.dropNaNs(0);
   * // return {
   * //    a: [0, 2, NaN, 4, 4], b: [0, 2, 3, NaN, 4],
   * //    c: [NaN, NaN, NaN, NaN,NaN]
   * // }
   *
   * // delete rows with atleast one NaN
   * df.dropNaNs(0, df.numColumns);
   * // returns empty df, since each row contains atleast one NaN
   *
   * // delete columns with all NaNs (default thresh=1)
   * df.dropNaNs(1);
   * // returns {a: [0, NaN, 2, NaN, 4, 4], b: [0, NaN, 2, 3, NaN, 4]}
   *
   * // delete columns with atleast one NaN
   * df.dropNaNs(1, df.numRows);
   * // returns empty df, since each column contains atleast one NaN
   *
   * ```
   */
  dropNaNs<R extends IndexType>(axis = 0, thresh = 1, subset?: (string&keyof T)[]|Series<R>):
    DataFrame<T> {
    if (axis == 0) {
      if (subset instanceof Series) {
        throw new Error(
          'for axis=0, expected \'subset\' to be one of {list of column_names, undefined(all columns)}');
      }
      return this._dropNaNsRows(thresh, subset);
    } else if (axis == 1) {
      if (subset instanceof Array) {
        throw new Error(
          'for axis=1, expected \'subset\' to be one of {Series<Integer> with indices to select rows, undefined(all rows)}');
      }
      return this._dropNaNsColumns(thresh, subset);
    } else {
      throw new Error('invalid axis value, expected {0, 1} ');
    }
  }

  /**
   * Compute the trigonometric sine for all NumericSeries in the DataFrame
   *
   * @returns A DataFrame with the operation performed on all NumericSeries
   * @example
   * ```typescript
   * import {DataFrame, Series, Int8}  from '@rapidsai/cudf';
   *
   * const df = new DataFrame({
   *  a: Series.new({type: new Int8, data: [-3, 0, 3]})
   * });
   * df.sin();
   * // return {
   * //    a: [0, 0, 0],
   * // }
   * ```
   */
  sin<P extends keyof T>(memoryResource?: MemoryResource) {
    const series_map = {} as SeriesMap<T>;
    this._accessor.names.forEach((name) => {
      const ser = this.get(name);
      _throwIfNonNumeric(ser.type, `sin`);
      series_map[name] = (ser as any).sin(memoryResource);
    });
    return new DataFrame(series_map) as T[P] extends Numeric ? DataFrame<T>: never;
  }

  /**
   * Compute the trigonometric cosine for all NumericSeries in the DataFrame
   *
   * @returns A DataFrame with the operation performed on all NumericSeries
   * @example
   * ```typescript
   * import {DataFrame, Series, Int8}  from '@rapidsai/cudf';
   *
   * const df = new DataFrame({
   *  a: Series.new({type: new Int8, data: [-3, 0, 3]})
   * });
   * df.cos();
   * // return {
   * //    a: [0, 1, 0],
   * // }
   * ```
   */
  cos<P extends keyof T>(memoryResource?: MemoryResource) {
    const series_map = {} as SeriesMap<T>;
    this._accessor.names.forEach((name) => {
      const ser = this.get(name);
      _throwIfNonNumeric(ser.type, `cos`);
      series_map[name] = (ser as any).cos(memoryResource);
    });
    return new DataFrame(series_map) as T[P] extends Numeric ? DataFrame<T>: never;
  }

  /**
   * Compute the trigonometric tangent for all NumericSeries in the DataFrame
   *
   * @returns A DataFrame with the operation performed on all NumericSeries
   * @example
   * ```typescript
   * import {DataFrame, Series, Int8}  from '@rapidsai/cudf';
   *
   * const df = new DataFrame({
   *  a: Series.new({type: new Int8, data: [-3, 0, 3]})
   * });
   * df.tan();
   * // return {
   * //    a: [0, 0, 0],
   * // }
   * ```
   */
  tan<P extends keyof T>(memoryResource?: MemoryResource) {
    const series_map = {} as SeriesMap<T>;
    this._accessor.names.forEach((name) => {
      const ser = this.get(name);
      _throwIfNonNumeric(ser.type, `tan`);
      series_map[name] = (ser as any).tan(memoryResource);
    });
    return new DataFrame(series_map) as T[P] extends Numeric ? DataFrame<T>: never;
  }

  /**
   * Compute the trigonometric sine inverse for all NumericSeries in the DataFrame
   *
   * @returns A DataFrame with the operation performed on all NumericSeries
   * @example
   * ```typescript
   * import {DataFrame, Series, Int8}  from '@rapidsai/cudf';
   *
   * const df = new DataFrame({
   *  a: Series.new({type: new Int8, data: [-3, 0, 3]})
   * });
   * df.asin();
   * // return {
   * //    a: [0, 0, 0],
   * // }
   * ```
   */
  asin<P extends keyof T>(memoryResource?: MemoryResource) {
    const series_map = {} as SeriesMap<T>;
    this._accessor.names.forEach((name) => {
      const ser = this.get(name);
      _throwIfNonNumeric(ser.type, `asin`);
      series_map[name] = (ser as any).asin(memoryResource);
    });
    return new DataFrame(series_map) as T[P] extends Numeric ? DataFrame<T>: never;
  }

  /**
   * Compute the trigonometric cosine inverse for all NumericSeries in the DataFrame
   *
   * @returns A DataFrame with the operation performed on all NumericSeries
   * @example
   * ```typescript
   * import {DataFrame, Series, Int8}  from '@rapidsai/cudf';
   *
   * const df = new DataFrame({
   *  a: Series.new({type: new Int8, data: [-3, 0, 3]})
   * });
   * df.acos();
   * // return {
   * //    a: [0, 1, 0],
   * // }
   * ```
   */
  acos<P extends keyof T>(memoryResource?: MemoryResource) {
    const series_map = {} as SeriesMap<T>;
    this._accessor.names.forEach((name) => {
      const ser = this.get(name);
      _throwIfNonNumeric(ser.type, `acos`);
      series_map[name] = (ser as any).acos(memoryResource);
    });
    return new DataFrame(series_map) as T[P] extends Numeric ? DataFrame<T>: never;
  }

  /**
   * Compute the trigonometric tangent inverse for all NumericSeries in the DataFrame
   *
   * @returns A DataFrame with the operation performed on all NumericSeries
   * @example
   * ```typescript
   * import {DataFrame, Series, Int8}  from '@rapidsai/cudf';
   *
   * const df = new DataFrame({
   *  a: Series.new({type: new Int8, data: [-3, 0, 3]})
   * });
   * df.atan();
   * // return {
   * //    a: [-1, 0, 1],
   * // }
   * ```
   */
  atan<P extends keyof T>(memoryResource?: MemoryResource) {
    const series_map = {} as SeriesMap<T>;
    this._accessor.names.forEach((name) => {
      const ser = this.get(name);
      _throwIfNonNumeric(ser.type, `atan`);
      series_map[name] = (ser as any).atan(memoryResource);
    });
    return new DataFrame(series_map) as T[P] extends Numeric ? DataFrame<T>: never;
  }

  /**
   * Compute the hyperbolic sine for all NumericSeries in the DataFrame
   *
   * @returns A DataFrame with the operation performed on all NumericSeries
   * @example
   * ```typescript
   * import {DataFrame, Series, Int8}  from '@rapidsai/cudf';
   *
   * const df = new DataFrame({
   *  a: Series.new({type: new Int8, data: [-3, 0, 3]})
   * });
   * df.sinh();
   * // return {
   * //    a: [-10, 0, 10],
   * // }
   * ```
   */
  sinh<P extends keyof T>(memoryResource?: MemoryResource) {
    const series_map = {} as SeriesMap<T>;
    this._accessor.names.forEach((name) => {
      const ser = this.get(name);
      _throwIfNonNumeric(ser.type, `sinh`);
      series_map[name] = (ser as any).sinh(memoryResource);
    });
    return new DataFrame(series_map) as T[P] extends Numeric ? DataFrame<T>: never;
  }

  /**
   * Compute the hyperbolic cosine for all NumericSeries in the DataFrame
   *
   * @returns A DataFrame with the operation performed on all NumericSeries
   * @example
   * ```typescript
   * import {DataFrame, Series, Int8}  from '@rapidsai/cudf';
   *
   * const df = new DataFrame({
   *  a: Series.new({type: new Int8, data: [-3, 0, 3]})
   * });
   * df.cosh();
   * // return {
   * //    a: [10, 1, 10],
   * // }
   * ```
   */
  cosh<P extends keyof T>(memoryResource?: MemoryResource) {
    const series_map = {} as SeriesMap<T>;
    this._accessor.names.forEach((name) => {
      const ser = this.get(name);
      _throwIfNonNumeric(ser.type, `cosh`);
      series_map[name] = (ser as any).cosh(memoryResource);
    });
    return new DataFrame(series_map) as T[P] extends Numeric ? DataFrame<T>: never;
  }

  /**
   * Compute the hyperbolic tangent for all NumericSeries in the DataFrame
   *
   * @returns A DataFrame with the operation performed on all NumericSeries
   * @example
   * ```typescript
   * import {DataFrame, Series, Int8}  from '@rapidsai/cudf';
   *
   * const df = new DataFrame({
   *  a: Series.new({type: new Int8, data: [-3, 0, 3]})
   * });
   * df.tanh();
   * // return {
   * //    a: [0, 0, 0],
   * // }
   * ```
   */
  tanh<P extends keyof T>(memoryResource?: MemoryResource) {
    const series_map = {} as SeriesMap<T>;
    this._accessor.names.forEach((name) => {
      const ser = this.get(name);
      _throwIfNonNumeric(ser.type, `tanh`);
      series_map[name] = (ser as any).tanh(memoryResource);
    });
    return new DataFrame(series_map) as T[P] extends Numeric ? DataFrame<T>: never;
  }

  /**
   * Compute the hyperbolic sine inverse for all NumericSeries in the DataFrame
   *
   * @returns A DataFrame with the operation performed on all NumericSeries
   * @example
   * ```typescript
   * import {DataFrame, Series, Int8}  from '@rapidsai/cudf';
   *
   * const df = new DataFrame({
   *  a: Series.new({type: new Int8, data: [-3, 0, 3]})
   * });
   * df.asinh();
   * // return {
   * //    a: [-1, 0, 1],
   * // }
   * ```
   */
  asinh<P extends keyof T>(memoryResource?: MemoryResource) {
    const series_map = {} as SeriesMap<T>;
    this._accessor.names.forEach((name) => {
      const ser = this.get(name);
      _throwIfNonNumeric(ser.type, `asinh`);
      series_map[name] = (ser as any).asinh(memoryResource);
    });
    return new DataFrame(series_map) as T[P] extends Numeric ? DataFrame<T>: never;
  }

  /**
   * Compute the hyperbolic cosine inverse for all NumericSeries in the DataFrame
   *
   * @returns A DataFrame with the operation performed on all NumericSeries
   * @example
   * ```typescript
   * import {DataFrame, Series, Int8}  from '@rapidsai/cudf';
   *
   * const df = new DataFrame({
   *  a: Series.new({type: new Int8, data: [-3, 0, 3]})
   * });
   * df.acosh();
   * // return {
   * //    a: [0, 0, 1],
   * // }
   * ```
   */
  acosh<P extends keyof T>(memoryResource?: MemoryResource) {
    const series_map = {} as SeriesMap<T>;
    this._accessor.names.forEach((name) => {
      const ser = this.get(name);
      _throwIfNonNumeric(ser.type, `acosh`);
      series_map[name] = (ser as any).acosh(memoryResource);
    });
    return new DataFrame(series_map) as T[P] extends Numeric ? DataFrame<T>: never;
  }

  /**
   * Compute the hyperbolic tangent inverse for all NumericSeries in the DataFrame
   *
   * @returns A DataFrame with the operation performed on all NumericSeries
   * @example
   * ```typescript
   * import {DataFrame, Series, Int8}  from '@rapidsai/cudf';
   *
   * const df = new DataFrame({
   *  a: Series.new({type: new Int8, data: [-3, 0, 3]})
   * });
   * df.atanh();
   * // return {
   * //    a: [0, 0, 0],
   * // }
   * ```
   */
  atanh<P extends keyof T>(memoryResource?: MemoryResource) {
    const series_map = {} as SeriesMap<T>;
    this._accessor.names.forEach((name) => {
      const ser = this.get(name);
      _throwIfNonNumeric(ser.type, `atanh`);
      series_map[name] = (ser as any).atanh(memoryResource);
    });
    return new DataFrame(series_map) as T[P] extends Numeric ? DataFrame<T>: never;
  }

  /**
   * Compute the exponential (base e, euler number) for all NumericSeries in the DataFrame
   *
   * @returns A DataFrame with the operation performed on all NumericSeries
   * @example
   * ```typescript
   * import {DataFrame, Series}  from '@rapidsai/cudf';
   *
   * const df = new DataFrame({
   *  a: Series.new([-1.2, 2.5])
   * });
   * df.exp();
   * // return {
   * //    a: [0.30119421191220214, 12.182493960703473],
   * // }
   * ```
   */
  exp<P extends keyof T>(memoryResource?: MemoryResource) {
    const series_map = {} as SeriesMap<T>;
    this._accessor.names.forEach((name) => {
      const ser = this.get(name);
      _throwIfNonNumeric(ser.type, `exp`);
      series_map[name] = (ser as any).exp(memoryResource);
    });
    return new DataFrame(series_map) as T[P] extends Numeric ? DataFrame<T>: never;
  }

  /**
   * Compute the natural logarithm (base e) for all NumericSeries in the DataFrame
   *
   * @returns A DataFrame with the operation performed on all NumericSeries
   * @example
   * ```typescript
   * import {DataFrame, Series}  from '@rapidsai/cudf';
   *
   * const df = new DataFrame({
   *  a: Series.new([-1.2, 2.5, 4])
   * });
   * df.log();
   * // return {
   * //    a: [NaN, 0.9162907318741551, 1.3862943611198906],
   * // }
   * ```
   */
  log<P extends keyof T>(memoryResource?: MemoryResource) {
    const series_map = {} as SeriesMap<T>;
    this._accessor.names.forEach((name) => {
      const ser = this.get(name);
      _throwIfNonNumeric(ser.type, `log`);
      series_map[name] = (ser as any).log(memoryResource);
    });
    return new DataFrame(series_map) as T[P] extends Numeric ? DataFrame<T>: never;
  }

  /**
   * Compute the square-root (x^0.5) for all NumericSeries in the DataFrame
   *
   * @returns A DataFrame with the operation performed on all NumericSeries
   * @example
   * ```typescript
   * import {DataFrame, Series}  from '@rapidsai/cudf';
   *
   * const df = new DataFrame({
   *  a: Series.new([-1.2, 2.5, 4])
   * });
   * df.sqrt();
   * // return {
   * //    a: [NaN, 1.5811388300841898, 2],
   * // }
   * ```
   */
  sqrt<P extends keyof T>(memoryResource?: MemoryResource) {
    const series_map = {} as SeriesMap<T>;
    this._accessor.names.forEach((name) => {
      const ser = this.get(name);
      _throwIfNonNumeric(ser.type, `sqrt`);
      series_map[name] = (ser as any).sqrt(memoryResource);
    });
    return new DataFrame(series_map) as T[P] extends Numeric ? DataFrame<T>: never;
  }

  /**
   * Compute the cube-root (x^(1.0/3)) for all NumericSeries in the DataFrame
   *
   * @returns A DataFrame with the operation performed on all NumericSeries
   * @example
   * ```typescript
   * import {DataFrame, Series}  from '@rapidsai/cudf';
   *
   * const df = new DataFrame({
   *  a: Series.new([-1.2, 2.5])
   * });
   * df.cbrt();
   * // return {
   * //    a: [-1.0626585691826111, 1.3572088082974534],
   * // }
   * ```
   */
  cbrt<P extends keyof T>(memoryResource?: MemoryResource) {
    const series_map = {} as SeriesMap<T>;
    this._accessor.names.forEach((name) => {
      const ser = this.get(name);
      _throwIfNonNumeric(ser.type, `cbrt`);
      series_map[name] = (ser as any).cbrt(memoryResource);
    });
    return new DataFrame(series_map) as T[P] extends Numeric ? DataFrame<T>: never;
  }

  /**
   * Compute the smallest integer value not less than arg for all NumericSeries in the DataFrame
   *
   * @returns A DataFrame with the operation performed on all NumericSeries
   * @example
   * ```typescript
   * import {DataFrame, Series}  from '@rapidsai/cudf';
   *
   * const df = new DataFrame({
   *  a: Series.new([-1.2, 2.5, -3, 4.6, 5])
   * });
   * df.ceil();
   * // return {
   * //    a: [-1, 3, -3, 5, 5],
   * // }
   * ```
   */
  ceil<P extends keyof T>(memoryResource?: MemoryResource) {
    const series_map = {} as SeriesMap<T>;
    this._accessor.names.forEach((name) => {
      const ser = this.get(name);
      _throwIfNonNumeric(ser.type, `ceil`);
      series_map[name] = (ser as any).ceil(memoryResource);
    });
    return new DataFrame(series_map) as T[P] extends Numeric ? DataFrame<T>: never;
  }

  /**
   * Compute the largest integer value not greater than arg for all NumericSeries in the DataFrame
   *
   * @returns A DataFrame with the operation performed on all NumericSeries
   * @example
   * ```typescript
   * import {DataFrame, Series}  from '@rapidsai/cudf';
   *
   * const df = new DataFrame({
   *  a: Series.new([-1.2, 2.5, -3, 4.6, 5])
   * });
   * df.floor();
   * // return {
   * //    a: [-2, 2, -3, 4, 5],
   * // }
   * ```
   */
  floor<P extends keyof T>(memoryResource?: MemoryResource) {
    const series_map = {} as SeriesMap<T>;
    this._accessor.names.forEach((name) => {
      const ser = this.get(name);
      _throwIfNonNumeric(ser.type, `floor`);
      series_map[name] = (ser as any).floor(memoryResource);
    });
    return new DataFrame(series_map) as T[P] extends Numeric ? DataFrame<T>: never;
  }

  /**
   * Compute the absolute value for all NumericSeries in the DataFrame
   *
   * @returns A DataFrame with the operation performed on all NumericSeries
   * @example
   * ```typescript
   * import {DataFrame, Series}  from '@rapidsai/cudf';
   *
   * const df = new DataFrame({
   *  a: Series.new([-1, 2, -3, 4, 5])
   * });
   * df.abs();
   * // return {
   * //    a: [1, 2, 3, 4, 5],
   * // }
   * ```
   */
  abs<P extends keyof T>(memoryResource?: MemoryResource) {
    const series_map = {} as SeriesMap<T>;
    this._accessor.names.forEach((name) => {
      const ser = this.get(name);
      _throwIfNonNumeric(ser.type, `abs`);
      series_map[name] = (ser as any).abs(memoryResource);
    });
    return new DataFrame(series_map) as T[P] extends Numeric ? DataFrame<T>: never;
  }

  /**
   * Compute the logical not (!) for all NumericSeries in the DataFrame
   *
   * @returns A DataFrame with the operation performed on all NumericSeries
   * @example
   * ```typescript
   * import {DataFrame, Series}  from '@rapidsai/cudf';
   *
   * const df = new DataFrame({
   *  a: Series.new([0, 1, 2, 3, 4])
   * });
   * df.not();
   * // return {
   * //    a: [true, false, false, false, false],
   * // }
   * ```
   */
  not<P extends keyof T>(memoryResource?: MemoryResource) {
    const series_map = {} as SeriesMap<T>;
    this._accessor.names.forEach((name) => {
      const ser = this.get(name);
      _throwIfNonNumeric(ser.type, `not`);
      series_map[name] = (ser as any).not(memoryResource);
    });
    return new DataFrame(series_map) as T[P] extends Numeric ? DataFrame<T>: never;
  }

  /**
   * Return a Series containing the unbiased kurtosis result for each Series in the
   * DataFrame.
   *
   * @param skipNulls Exclude NA/null values. If an entire row/column is NA, the result will be NA.
   * @returns A Series containing the unbiased kurtosis result for all Series in the DataFrame
   * @example
   * ```typescript
   * import {DataFrame, Series}  from '@rapidsai/cudf';
   *
   * const df = new DataFrame({
   *  a: Series.new([1, 2, 3, 4]),
   *  b: Series.new([7, 8, 9, 10])
   * });
   * df.kurtosis(); // {-1.1999999999999904, -1.2000000000000686}
   * ```
   */
  kurtosis<P extends keyof T>(skipNulls = true) {
    const result = this.names.map((name) => {
      const ser = this.get(name);
      _throwIfNonNumeric(ser.type, `kurtosis`);
      return (this.get(name) as any).kurtosis(skipNulls);
    });
    return Series.new(result) as any as Series < T[P] extends Numeric ? Numeric : never > ;
  }

  /**
   * Return a Series containing the unbiased skew result for each Series in the
   * DataFrame.
   *
   * @param skipNulls Exclude NA/null values. If an entire row/column is NA, the result will be NA.
   * @returns A Series containing the unbiased skew result for all Series in the DataFrame
   * @example
   * ```typescript
   * import {DataFrame, Series}  from '@rapidsai/cudf';
   *
   * const df = new DataFrame({
   *  a: Series.new([1, 2, 3, 4, 5, 6, 6]),
   *  b: Series.new([7, 8, 9, 10, 11, 12, 12])
   * });
   * df.skew(); // {-0.288195490292614, -0.2881954902926153}
   * ```
   */
  skew<P extends keyof T>(skipNulls = true) {
    const result = this.names.map((name) => {
      const ser = this.get(name);
      _throwIfNonNumeric(ser.type, `skew`);
      return (this.get(name) as any).skew(skipNulls);
    });
    return Series.new(result) as any as Series < T[P] extends Numeric ? Numeric : never > ;
  }

  /**
   * Compute the sum for all Series in the DataFrame.
   *
   * @param subset List of columns to select (all columns are considered by
   * default).
   * @param skipNulls The optional skipNulls if true drops NA and null values before computing
   *   reduction,
   * else if skipNulls is false, reduction is computed directly.
   * @param memoryResource Memory resource used to allocate the result Column's device memory.
   *
   * @returns A Series containing the sum of all values for each Series
   * @example
   * ```typescript
   * import {DataFrame, Series}  from '@rapidsai/cudf';
   *
   * const df = new DataFrame({
   *  a: Series.new([1, 2]),
   *  b: Series.new([3.5, 4])
   * });
   * df.sum(); // [3, 7.5]
   *
   * const df2 = new DataFrame({
   *  a: Series.new(['foo', 'bar']),
   *  b: Series.new([3, 4])
   * });
   *
   * df2.sum(); // returns `never`
   * ```
   */
  sum<P extends keyof T = keyof T>(subset?: readonly P[],
                                   skipNulls = true,
                                   memoryResource?: MemoryResource) {
    subset = (subset == undefined) ? this.names as readonly P[] : subset;
    const containsAllFloatingPoint =
      subset.every((name) => FloatTypes.some((t) => compareTypes(t, this.types[name])));
    const containsAllIntegral =
      subset.every((name) => IntegralTypes.some((t) => compareTypes(t, this.types[name])));
    if (!(containsAllFloatingPoint !== containsAllIntegral)) {
      throw new TypeError(
        `sum operation requires dataframe to be entirely of dtype FloatingPoint OR Integral.`);
    }

    const sums = subset.map((name) => (this.get(name) as any).sum(skipNulls, memoryResource));

    return (containsAllIntegral ? Series.new({type: new Int64, data: sums}) : Series.new(sums)) as (
             Series < T[P] extends Integral ? T[P] extends FloatingPoint ? never : Integral
                                            : T[P] extends FloatingPoint ? FloatingPoint : never >);
  }

  /**
   * Convert NaNs (if any) to nulls.
   *
   * @param subset List of float columns to consider to replace NaNs with nulls.
   *
   * @returns DataFrame<T> with NaNs(if any) converted to nulls
   *
   * @example
   * ```typescript
   * import {DataFrame, Series, Int32, Float32}  from '@rapidsai/cudf';
   * const df = new DataFrame({
   *  a: Series.new({type: new Int32, data: [0, 1, 2, 3, 4, 4]}),
   *  b: Series.new({type: new Float32, data: [0, NaN, 2, 3, 4, 4]})
   * });
   * df.get("b").nullCount; // 0
   * const df1 = df.nansToNulls();
   * df1.get("b").nullCount; // 1
   *
   * ```
   */
  nansToNulls(subset?: (keyof T)[], memoryResource?: MemoryResource): DataFrame<T> {
    subset           = (subset == undefined) ? this.names as (keyof T)[] : subset;
    const temp       = new Table({columns: this.select(subset)._accessor.columns});
    const series_map = {} as SeriesMap<T>;
    this._accessor.names.forEach((name, index) => {
      if (FloatTypes.some((t) => compareTypes(this.get(name).type, t))) {
        series_map[name] = Series.new(temp.getColumnByIndex(index).nansToNulls(memoryResource));
      } else {
        series_map[name] = Series.new(temp.getColumnByIndex(index));
      }
    });
    return new DataFrame(series_map);
  }

  /**
   * Creates a DataFrame replacing any FloatSeries with a Bool8Series where `true` indicates the
   * value is `NaN` and `false` indicates the value is valid.
   *
   * @returns a DataFrame replacing instances of FloatSeries with a Bool8Series where `true`
   * indicates the value is `NaN`
   *
   * @example
   * ```typescript
   * import {DataFrame, Series, Int32, Float32}  from '@rapidsai/cudf';
   * const df = new DataFrame({
   *  a: Series.new({type: new Int32, data: [0, 1, null]}),
   *  b: Series.new({type: new Float32, data: [0, NaN, 2]})
   * });
   *
   * df.isNaN()
   * // return {
   * //    a: [0, 1, null],
   * //    b: [false, true, false],
   * // }
   * ```
   */
  isNaN(memoryResource?: MemoryResource): DataFrame<T> {
    return new DataFrame(
      this.names.reduce((map, name) => ({
                          ...map,
                          [name]: FloatTypes.some((t) => compareTypes(this.get(name).type, t))
                                    ? Series.new(this._accessor.get(name).isNaN(memoryResource))
                                    : Series.new(this._accessor.get(name))
                        }),
                        {} as SeriesMap<T>));
  }

  /**
   * Creates a DataFrame of `BOOL8` Series where `true` indicates the value is null and
   * `false` indicates the value is valid.
   *
   * @returns a DataFrame containing Series of 'BOOL8' where 'true' indicates the value is null
   *
   * @example
   * ```typescript
   * import {DataFrame, Series}  from '@rapidsai/cudf';
   * const df = new DataFrame({
   *  a: Series.new([0, null, 2]),
   *  b: Series.new(['foo', 'bar', null])
   * });
   *
   * df.isNull()
   * // return {
   * //    a: [false, true, false],
   * //    b: [false, false, true],
   * // }
   * ```
   */
  isNull(memoryResource?: MemoryResource): DataFrame<{[P in keyof T]: Bool8}> {
    return new DataFrame(
      this.names.reduce((cols, name) => ({...cols, [name]: this.get(name).isNull(memoryResource)}),
                        {} as SeriesMap<{[P in keyof T]: Bool8}>));
  }

  /**
   * Creates a DataFrame replacing any FloatSeries with a Bool8Series where `false` indicates the
   * value is `NaN` and `true` indicates the value is valid.
   *
   * @returns a DataFrame replacing instances of FloatSeries with a Bool8Series where `false`
   * indicates the value is `NaN`
   *
   * @example
   * ```typescript
   * import {DataFrame, Series, Int32, Float32}  from '@rapidsai/cudf';
   * const df = new DataFrame({
   *  a: Series.new({type: new Int32, data: [0, 1, null]}),
   *  b: Series.new({type: new Float32, data: [0, NaN, 2]})
   * });
   *
   * df.isNotNaN()
   * // return {
   * //    a: [0, 1, null],
   * //    b: [true, false, true],
   * // }
   * ```
   */
  isNotNaN(): DataFrame<T> {
    return new DataFrame(
      this.names.reduce((map, name) => ({
                          ...map,
                          [name]: FloatTypes.some((t) => compareTypes(this.get(name).type, t))
                                    ? Series.new(this._accessor.get(name).isNotNaN())
                                    : Series.new(this._accessor.get(name))
                        }),
                        {} as SeriesMap<T>));
  }

  /**
   * Creates a DataFrame of `BOOL8` Series where `false` indicates the value is null and
   * `true` indicates the value is valid.
   *
   * @returns a DataFrame containing Series of 'BOOL8' where 'false' indicates the value is null
   *
   * @example
   * ```typescript
   * import {DataFrame, Series}  from '@rapidsai/cudf';
   * const df = new DataFrame({
   *  a: Series.new([0, null, 2]),
   *  b: Series.new(['foo', 'bar', null])
   * });
   *
   * df.isNotNull()
   * // return {
   * //    a: [true, false, true],
   * //    b: [true, true, false],
   * // }
   * ```
   */
  isNotNull(): DataFrame<{[P in keyof T]: Bool8}> {
    return new DataFrame(
      this.names.reduce((cols, name) => ({...cols, [name]: this.get(name).isNotNull()}),
                        {} as SeriesMap<{[P in keyof T]: Bool8}>));
  }

  /**
   * Replace null values with a value.
   *
   * @param value The scalar value to use in place of nulls.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   *
   * @example
   * ```typescript
   * import {DataFrame, Series} from '@rapidsai/cudf';
   *
   * const df = new DataFrame({
   *  a: Series.new([0, null, 2]);
   *  b: Series.new([null, null, null]);
   * });
   *
   * df.replaceNulls(1);
   * // return {
   * //    a: [0, 1, 2],
   * //    b: [1, 1, 1],
   * // }
   * ```
   */
  replaceNulls<R extends DataType>(value: R['scalarType'],
                                   memoryResource?: MemoryResource): DataFrame<T>;

  /**
   * Replace null values with the corresponding elements from another Map of Series.
   *
   * @param value The map of Series to use in place of nulls.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   *
   * @example
   * ```typescript
   * import {DataFrame, Series} from '@rapidsai/cudf';
   *
   * const df = new DataFrame({
   *  a: Series.new([0, null, 2]);
   *  b: Series.new([null, null, null]);
   * });
   *
   * df.replaceNulls({'a': Series.new([0, 1, 2]), 'b': Series.new([1, 1, 1])});
   * // return {
   * //    a: [0, 1, 2],
   * //    b: [1, 1, 1],
   * // }
   * ```
   */
  replaceNulls(value: SeriesMap<T>, memoryResource?: MemoryResource): DataFrame<T>;

  replaceNulls<R extends DataType>(value: SeriesMap<T>|R['scalarType'],
                                   memoryResource?: MemoryResource): DataFrame<T> {
    if (value instanceof Object) {
      const columns = new ColumnAccessor(_seriesToColumns(value));
      return new DataFrame(this.names.reduce(
        (map, name) => ({
          ...map,
          [name]: columns.names.includes(name) ? Series.new(this._accessor.get(name).replaceNulls(
                                                   columns.get(name), memoryResource))
                                               : Series.new(this._accessor.get(name))
        }),
        {} as SeriesMap<T>));
    } else {
      return new DataFrame(this.names.reduce(
        (map, name) => ({
          ...map,
          [name]: Series.new(this._accessor.get(name).replaceNulls(
            new Scalar({type: this._accessor.get(name).type, value: value}), memoryResource))
        }),
        {} as SeriesMap<T>));
    }
  }

  /**
   * Drops duplicate rows from a DataFrame
   *
   * @param keep Determines whether to keep the first, last, or none of the duplicate items.
   * @param nullsEqual Determines whether nulls are handled as equal values.
   * @param nullsFirst Determines whether null values are inserted before or after non-null
   *   values.
   * @param subset List of columns to consider when dropping rows (all columns are considered by
   * default).
   * @param memoryResource Memory resource used to allocate the result Column's device memory.
   *
   * @returns a DataFrame without duplicate rows
   * ```
   */
  dropDuplicates(keep: keyof typeof DuplicateKeepOption,
                 nullsEqual: boolean,
                 nullsFirst: boolean,
                 subset = this.names,
                 memoryResource?: MemoryResource) {
    const column_indices: number[] = [];
    const allNames                 = this.names;

    subset.forEach((col) => {
      if (allNames.includes(col)) {
        column_indices.push(allNames.indexOf(col));
      } else {
        throw new Error(`Unknown column name: ${col}`);
      }
    });
    const table = this.asTable().dropDuplicates(
      column_indices, DuplicateKeepOption[keep], nullsEqual, nullsFirst, memoryResource);
    return new DataFrame(
      allNames.reduce((map, name, i) => ({...map, [name]: Series.new(table.getColumnByIndex(i))}),
                      {} as SeriesMap<T>));
  }
}
