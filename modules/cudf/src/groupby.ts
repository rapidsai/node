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

import {MemoryResource} from '@nvidia/rmm';

import CUDF from './addon';
import {Column} from './column';
import {DataFrame, SeriesMap} from './data_frame';
import {Series} from './series';
import {Table} from './table';
import {NullOrder} from './types/enums'
import {TypeMap} from './types/mappings'

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
  offsets: number[],
  values?: DataFrame<ValuesMap>,
}

interface GroupbyConstructor {
  readonly prototype: CudfGroupBy;
  new(props: CudfGroupByProps): CudfGroupBy;
}

interface CudfGroupBy {
  _getGroups(values?: Table, memoryResource?: MemoryResource): any;

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
  getGroups(memoryResource?: MemoryResource): Groups<Pick<T, R>, Omit<T, R>> {
    const table      = this._values.asTable();
    const results    = this._getGroups(table, memoryResource);
    const series_map = {} as any;

    this._by.forEach(
      (name, index) => { series_map[name] = Series.new(results.keys.getColumnByIndex(index)); });
    results.keys = new DataFrame(series_map);

    if (results.values !== undefined) {
      this._values.names.forEach(
        (name,
         index) => { series_map[name] = Series.new(results.values.getColumnByIndex(index)); });
      results.values = new DataFrame(series_map);
    }

    return results;
  }

  protected prepare_results(results: {keys: Table, cols: Column[]}) {
    const {keys, cols} = results;

    const series_map = {} as SeriesMap<T>;
    this._by.forEach(
      (name, index) => { series_map[name] = Series.new(keys.getColumnByIndex(index)); });
    this._values.names.forEach((name, index) => { series_map[name] = Series.new(cols[index]); });
    return new DataFrame(series_map);
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
  // quantile(q = 0.5, interpolation = 'linear', memoryResource?: MemoryResource) {
  //   return this.prepare_results(
  //     this._quantile(q, this._values.asTable(), interpolation, memoryResource));
  // }
}
