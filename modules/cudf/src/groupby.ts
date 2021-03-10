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

import {Series, Struct} from '@nvidia/cudf';
import {MemoryResource} from '@nvidia/rmm';
import {Field} from 'apache-arrow';

import CUDF from './addon';
import {Column} from './column';
import {DataFrame, SeriesMap} from './data_frame';
import {Table} from './table';
import {DataType} from './types/dtypes';
import {NullOrder} from './types/enums'
import {Interpolation, TypeMap} from './types/mappings'

/*
 * @param keys DataFrame whose rows act as the groupby keys
 * @param include__nulls Indicates whether rows in `keys` that contain
 * NULL values should be included
 * @param keys_are_sorted Indicates whether rows in `keys` are already sorted
 * @param column_order If `keys_are_sorted == YES`, indicates whether each
 * column is ascending/descending. If empty, assumes all  columns are
 * ascending. Ignored if `keys_are_sorted == false`.
 * @param null_precedence If `keys_are_sorted == YES`, indicates the ordering
 * of null values in each column. Else, ignored. If empty, assumes all columns
 * use `null_order::BEFORE`. Ignored if `keys_are_sorted == false`.
 */

type SeriesMapOf<T extends string, R extends DataType> = {
  [P in T]: Series<R>
};

type Join<T extends unknown[], D extends string> =
  T extends [] ? '' : T extends [string | number | boolean | bigint]
                                  ? `${T[0]}`
                                  : T extends [string | number | boolean | bigint, ...infer U]
                                                ? `${T[0]}${D}${Join<U, D>}`
                                                : string;

export type GroupByProps<T extends TypeMap, R extends keyof T> = {
  obj: DataFrame<T>,
  by: R[],
  include_nulls?: boolean,
  keys_are_sorted?: boolean,
  column_order?: boolean[],
  null_precedence?: NullOrder[],
};

type CudfGroupByProps = {
  keys: Table,
  include_nulls?: boolean,
  keys_are_sorted?: boolean,
  column_order?: boolean[],
  null_precedence?: NullOrder[],
};

export type Groups<KeysMap extends TypeMap, ValuesMap extends TypeMap> = {
  keys: DataFrame<KeysMap>,
  offsets: Int32Array,
  values?: DataFrame<ValuesMap>,
}

interface GroupbyConstructor {
  readonly prototype: CudfGroupBy;
  new(props: CudfGroupByProps): CudfGroupBy;
}

interface CudfGroupBy {
  _getGroups(values?: Table,
             memoryResource?: MemoryResource): {keys: Table, offsets: Int32Array, values?: Table};

  _argmax(values: Table, memoryResource?: MemoryResource): {keys: Table, cols: Column[]};
  _argmin(values: Table, memoryResource?: MemoryResource): {keys: Table, cols: Column[]};
  _count(values: Table, memoryResource?: MemoryResource): {keys: Table, cols: Column[]};
  _max(values: Table, memoryResource?: MemoryResource): {keys: Table, cols: Column[]};
  _mean(values: Table, memoryResource?: MemoryResource): {keys: Table, cols: Column[]};
  _median(values: Table, memoryResource?: MemoryResource): {keys: Table, cols: Column[]};
  _min(values: Table, memoryResource?: MemoryResource): {keys: Table, cols: Column[]};
  _nth(n: number, values: Table, memoryResource?: MemoryResource): {keys: Table, cols: Column[]};
  _nunique(values: Table, memoryResource?: MemoryResource): {keys: Table, cols: Column[]};
  _std(values: Table, memoryResource?: MemoryResource): {keys: Table, cols: Column[]};
  _sum(values: Table, memoryResource?: MemoryResource): {keys: Table, cols: Column[]};
  _var(values: Table, memoryResource?: MemoryResource): {keys: Table, cols: Column[]};
  _quantile(q: number, values: Table, interpolation?: number, memoryResource?: MemoryResource):
    {keys: Table, cols: [Column]};
}

export class GroupBy<T extends TypeMap, R extends keyof T> extends(
  <GroupbyConstructor>CUDF.GroupBy) {
  private _by: R[];
  private _values: DataFrame<Omit<T, R>>;

  constructor(props: GroupByProps<T, R>) {
    const table = props.obj.select(props.by).asTable();
    const {
      include_nulls   = false,
      keys_are_sorted = false,
      column_order    = [],
      null_precedence = []
    } = props;

    const cudf_props: CudfGroupByProps = {
      keys: table,
      include_nulls: include_nulls,
      keys_are_sorted: keys_are_sorted,
      column_order: column_order,
      null_precedence: null_precedence,
    };
    super(cudf_props);
    this._by     = props.by;
    this._values = props.obj.drop(props.by);
  }

  /**
   * Return the Groups for this GroupBy
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  getGroups(memoryResource?: MemoryResource) {
    const {keys, offsets, values} = this._getGroups(this._values.asTable(), memoryResource);

    const results = {
      offsets,
      keys: new DataFrame(
        this._by.reduce((keys_map, name, index) =>
                          ({...keys_map, [name]: Series.new(keys.getColumnByIndex(index))}),
                        {} as SeriesMap<Pick<T, R>>))
    } as Groups<Pick<T, R>, Omit<T, R>>;

    if (values !== undefined) {
      results.values = new DataFrame(this._values.names.reduce(
        (values_map, name, index) =>
          ({...values_map, [name]: Series.new(values.getColumnByIndex(index))}),
        {} as SeriesMap<Omit<T, R>>));
    }

    return results;
  }

  protected prepare_results(results: {keys: Table, cols: Column[]}) {
    if (this._by.length == 1) {
      return this.prepare_results_single(results);
    } else {
      return this.prepare_results_multiple(results);
    }
  }

  protected prepare_results_single(results: {keys: Table, cols: Column[]}) {
    const {keys, cols} = results;
    const series_map   = {} as SeriesMap<T>;

    this._values.names.forEach((name, index) => { series_map[name] = Series.new(cols[index]); });
    series_map[this._by[0]] = Series.new(keys.getColumnByIndex(0));

    return new DataFrame(series_map);
  }

  protected prepare_results_multiple(results: {keys: Table, cols: Column[]}) {
    const {keys, cols} = results;

    type Subset   = Pick<T, R>;
    type Index    = Struct<Subset>;
    type IndexKey = Join<(keyof Subset)[], '_'>;
    type IndexMap = SeriesMapOf<IndexKey, Index>;

    type RestMap = SeriesMap<Omit<T, R>>;

    const rest_map = this._values.names.reduce(
      (xs, key, index) => ({...xs, [key]: Series.new(cols[index])}), {} as RestMap);

    const byname: keyof IndexMap = this._by.join('_');

    if (byname in rest_map) {
      throw new Error(`Groupby column name ${byname} already
      exists`);
    }

    const fields   = [];
    const children = [];
    for (const [index, name] of this._by.entries()) {
      const child = keys.getColumnByIndex(index)
      fields.push(Field.new({name: name as string, type: child.type}));
      children.push(Series.new(child));
    }

    const index = Series.new<Index>({type: new Struct(fields), children: children});

    return new DataFrame({[byname]: index, ...rest_map});
  }

  /**
   * Compute the index of the maximum value in each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  argmax(memoryResource?: MemoryResource) {
    return this.prepare_results(this._argmax(this._values.asTable(), memoryResource));
  }

  /**
   * Compute the index of the minimum value in each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  argmin(memoryResource?: MemoryResource) {
    return this.prepare_results(this._argmin(this._values.asTable(), memoryResource));
  }

  /**
   * Compute the size of each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  count(memoryResource?: MemoryResource) {
    return this.prepare_results(this._count(this._values.asTable(), memoryResource));
  }

  /**
   * Compute the maximum value in each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  max(memoryResource?: MemoryResource) {
    return this.prepare_results(this._max(this._values.asTable(), memoryResource));
  }

  /**
   * Compute the average value each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  mean(memoryResource?: MemoryResource) {
    return this.prepare_results(this._mean(this._values.asTable(), memoryResource));
  }

  /**
   * Compute the median value in each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  median(memoryResource?: MemoryResource) {
    return this.prepare_results(this._median(this._values.asTable(), memoryResource));
  }

  /**
   * Compute the minimum value in each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  min(memoryResource?: MemoryResource) {
    return this.prepare_results(this._min(this._values.asTable(), memoryResource));
  }

  /**
   * Return the nth value from each group
   *
   * @param n the index of the element to return
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  nth(n: number, memoryResource?: MemoryResource) {
    return this.prepare_results(this._nth(n, this._values.asTable(), memoryResource));
  }

  /**
   * Compute the number of unique values in each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  nunique(memoryResource?: MemoryResource) {
    return this.prepare_results(this._nunique(this._values.asTable(), memoryResource));
  }

  /**
   * Compute the standard deviation for each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  std(memoryResource?: MemoryResource) {
    return this.prepare_results(this._std(this._values.asTable(), memoryResource));
  }

  /**
   * Compute the sum of values in each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  sum(memoryResource?: MemoryResource) {
    return this.prepare_results(this._sum(this._values.asTable(), memoryResource));
  }

  /**
   * Compute the variance for each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  var(memoryResource?: MemoryResource) {
    return this.prepare_results(this._var(this._values.asTable(), memoryResource));
  }

  /**
   * Return values at the given quantile.
   *
   * @param q the quantile to compute, 0 <= q <= 1
   * @param interpolation This optional parameter specifies the interpolation method to use,
   *  when the desired quantile lies between two data points i and j.
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  quantile(q                                         = 0.5,
           interpolation: keyof typeof Interpolation = 'linear',
           memoryResource?: MemoryResource) {
    return this.prepare_results(
      this._quantile(q, this._values.asTable(), Interpolation[interpolation], memoryResource));
  }
}
