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

import {MemoryResource} from '@rapidsai/rmm';
import * as arrow from 'apache-arrow';
import {Readable} from 'stream';

import {Column} from './column';
import {ColumnAccessor} from './column_accessor';
import {Join, JoinResult} from './dataframe/join';
import {GroupByMultiple, GroupByMultipleProps, GroupBySingle, GroupBySingleProps} from './groupby';
import {AbstractSeries, Float32Series, Float64Series, Series} from './series';
import {Table, ToArrowMetadata} from './table';
import {CSVToCUDFType, CSVTypeMap, ReadCSVOptions, WriteCSVOptions} from './types/csv';
import {Bool8, DataType, IndexType, Numeric} from './types/dtypes';
import {NullOrder} from './types/enums';
import {ColumnsMap, CommonType, TypeMap} from './types/mappings';

export type SeriesMap<T extends TypeMap> = {
  [P in keyof T]: AbstractSeries<T[P]>
};

export type OrderSpec = {
  ascending: boolean,
  null_order: NullOrder
};

type JoinType = 'inner'|'outer'|'left'|'right'|'leftsemi'|'leftanti';

type JoinProps<
  Rhs extends TypeMap,
  TOn extends string,
  How extends JoinType = 'inner',
  LSuffix extends string = '',
  RSuffix extends string = '',
> = {
  other: DataFrame<Rhs>,
  on: TOn[],
  how: How,
  lsuffix?: LSuffix,
  rsuffix?: RSuffix,
  nullEquality?: boolean,
};

type CombinedGroupByProps<T extends TypeMap, R extends keyof T, IndexKey extends string> =
  GroupBySingleProps<T, R>|Partial<GroupByMultipleProps<T, R, IndexKey>>;

function _seriesToColumns<T extends TypeMap>(data: SeriesMap<T>) {
  const columns = {} as any;
  for (const [name, series] of Object.entries(data)) { columns[name] = series._col; }
  return <ColumnsMap<T>>columns;
}

/**
 * A GPU Dataframe object.
 */
export class DataFrame<T extends TypeMap = any> {
  /**
   * Read a csv from disk and create a cudf.DataFrame
   *
   * @example
   * ```typescript
   * import {DataFrame, Series}  from '@rapidsai/cudf';
   * const df = DataFrame.readCSV({
   *  header: 0,
   *  sourceType: 'files',
   *  sources: ['test.csv'],
   *  dataTypes: {
   *    a: 'int16',
   *    b: 'bool',
   *    c: 'float32',
   *    d: 'str'
   *  }
   * })
   * ```
   */
  public static readCSV<T extends CSVTypeMap = any>(options: ReadCSVOptions<T>) {
    const {names, table} = Table.readCSV(options);
    return new DataFrame(new ColumnAccessor(
      names.reduce((map, name, i) => ({...map, [name]: table.getColumnByIndex(i)}),
                   {} as ColumnsMap<{[P in keyof T]: CSVToCUDFType<T[P]>}>)));
  }

  private _accessor: ColumnAccessor<T>;

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
  constructor(data: ColumnAccessor<T>|SeriesMap<T>) {
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
  get numRows() { return this._accessor.columns[0].length; }

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

  /** @ignore */
  asTable() { return new Table({columns: this._accessor.columns}); }

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
  select<R extends keyof T>(names: R[]) {
    return new DataFrame(this._accessor.selectByColumnNames(names));
  }

  /**
   * Return a new DataFrame with new columns added.
   *
   * @param data mapping of names to new columns to add
   *
   * @example
   * ```typescript
   * import {DataFrame} from '@rapidsai/cudf';
   *
   * const df = new DataFrame({a: [1, 2, 3]});
   *
   * df.assign({b: ["foo", "bar", "bar"]})
   * // returns df {a: [1, 2, 3], b: ["foo", "bar", "bar"]}
   * ```
   */
  assign<R extends TypeMap>(data: SeriesMap<R>): DataFrame<T&R>;

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
  assign<R extends TypeMap>(data: DataFrame<R>): DataFrame<T&R>;

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
  drop<R extends keyof T>(names: R[]) { return new DataFrame(this._accessor.dropColumns(names)); }

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
   * Casts each Series in this DataFrame to a new dtype (similar to `static_cast` in C++).
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
   *
   * @example
   * ```typescript
   * import {DataFrame, Series, Int32, Float32}  from '@rapidsai/cudf';
   * const df = new DataFrame({
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
   * df.orderBy({a: {ascending: true, null_order: NullOrder.BEFORE}});
   * // Int32Series [0, 5, 4, 3, 2, 1]
   *
   * df.orderBy({a: {ascending: true, null_order: NullOrder.AFTER}});
   * // Int32Series [5, 4, 3, 2, 1, 0]
   *
   * df.orderBy({a: {ascending: false, null_order: NullOrder.BEFORE}});
   * // Int32Series [1, 2, 3, 4, 5, 0]
   *
   * df.orderBy({a: {ascending: false, null_order: NullOrder.AFTER}});
   * // Int32Series [0, 1, 2, 3, 4, 5]
   * ```
   */
  orderBy<R extends keyof T>(options: {[P in R]: OrderSpec}) {
    const column_orders = new Array<boolean>();
    const null_orders   = new Array<NullOrder>();
    const columns       = new Array<Column<T[keyof T]>>();
    const entries       = Object.entries(options) as [R, OrderSpec][];
    entries.forEach(([name, {ascending, null_order}]) => {
      const child = this.get(name);
      if (child) {
        columns.push(child._col as Column<T[keyof T]>);
        column_orders.push(ascending);
        null_orders.push(null_order);
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
   *   Default: AFTER
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
   * df.sortValues({a: {ascending: true, null_order: NullOrder.AFTER}})
   * // {a: [0, 1, 2, 3, 4, null], b: [5, 4, 3, 2, 1, 0]}
   *
   * df.sortValues({a: {ascending: true, null_order: NullOrder.BEFORE}})
   * // {a: [null, 0, 1, 2, 3, 4], b: [0, 5, 4, 3, 2, 1]}
   *
   * df.sortValues({a: {ascending: false, null_order: NullOrder.AFTER}})
   * // {a: [4, 3, 2, 1, 0, null], b: [1, 2, 3, 4, 5, 0]}
   *
   * df.sortValues({a: {ascending: false, null_order: NullOrder.BEFORE}})
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
      R[P] extends Numeric ? P extends TOn //
        ? CommonType<T[P], Numeric & R[P]> //
        : JoinResult<T, R, TOn, LSuffix, RSuffix>[P] //
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
  _dropNullsRows(thresh = 1, subset = this.names) {
    const column_names: (keyof T)[] = [];
    const column_indices: number[]  = [];
    subset.forEach((col, idx) => {
      if (this.names.includes(col)) {
        column_names.push(col);
        column_indices.push(idx);
      } else {
        throw new Error(`Unknown column name: ${col.toString()}`);
      }
    });

    const table_result = new Table({columns: this._accessor.columns});
    const result       = table_result.drop_nulls(column_indices, thresh);
    return new DataFrame(this.names.reduce(
      (map, name, i) => ({...map, [name]: Series.new(result.getColumnByIndex(i))}),
      {} as SeriesMap<T>));
  }
  /**
   * drop rows with NaN values (float type only)
   * @ignore
   */
  _dropNaNsRows(thresh = 1, subset = this.names) {
    const column_names: (keyof T)[] = [];
    const column_indices: number[]  = [];
    subset.forEach((col, idx) => {
      if (this.names.includes(col) &&
          (this.get(col) instanceof Float32Series || this.get(col) instanceof Float64Series)) {
        column_names.push(col);
        column_indices.push(idx);
      } else if (!this.names.includes(col)) {
        throw new Error(`Unknown column name: ${col.toString()}`);
      } else {
        // col exists but not of floating type
        thresh -= 1;
      }
    });
    const table_result = new Table({columns: this._accessor.columns});
    const result       = table_result.drop_nans(column_indices, thresh);
    return new DataFrame(this.names.reduce(
      (map, name, i) => ({...map, [name]: Series.new(result.getColumnByIndex(i))}),
      {} as SeriesMap<T>));
  }
  /**
   * drop columns with nulls
   * @ignore
   */
  _dropNullsColumns(thresh = 1, subset?: Series) {
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
  _dropNaNsColumns(thresh = 1, subset?: Series, memoryResource?: MemoryResource) {
    const column_names: (keyof T)[] = [];
    const df                        = (subset !== undefined) ? this.gather(subset) : this;

    this.names.forEach(col => {
      if (df.get(col) instanceof Float32Series || df.get(col) instanceof Float64Series) {
        const nanCount =
          df.get(col)._col.nans_to_nulls(memoryResource).nullCount - this.get(col).nullCount;

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
  nansToNulls(subset?: (keyof T)[]): DataFrame<T> {
    subset           = (subset == undefined) ? this.names as (keyof T)[] : subset;
    const temp       = new Table({columns: this.select(subset)._accessor.columns});
    const series_map = {} as SeriesMap<T>;
    this._accessor.names.forEach((name, index) => {
      if (this.get(name) instanceof Float32Series || this.get(name) instanceof Float64Series) {
        series_map[name] = Series.new(temp.getColumnByIndex(index).nans_to_nulls());
      } else {
        series_map[name] = Series.new(temp.getColumnByIndex(index));
      }
    });
    return new DataFrame(series_map);
  }

  /**
   * Creates a DataFrame of `BOOL8` Series where `true` indicates the value is `null` or `NaN` and
   * `false` indicates the value is valid.
   *
   * @returns DataFrame<Bool8> with series of `BOOL8` elements with `true` representing `null` or
   * `NaN` values
   *
   * @example
   * ```typescript
   * import {DataFrame, Series, Int32, Float32}  from '@rapidsai/cudf';
   * const df = new DataFrame({
   *  a: Series.new({type: new Int32, data: [0, 1, 2, 3, 4, 4]}),
   *  b: Series.new({type: new Float32, data: [0, NaN, 2, 3, 4, 4]})
   * });
   *
   * df.isNull()
   * ```
   */
  isNull(): DataFrame<T> {
    const temp       = new Table({columns: this.select(this.names)._accessor.columns});
    const series_map = {} as SeriesMap<T>;
    this._accessor.names.forEach((name, index) => {
      if (this.get(name) instanceof Float32Series || this.get(name) instanceof Float64Series) {
        series_map[name] = Series.new(temp.getColumnByIndex(index).isNaN()) as any;
      } else {
        series_map[name] = Series.new(temp.getColumnByIndex(index).isNull()) as any;
      }
    });
    return new DataFrame(series_map);
  }
}
