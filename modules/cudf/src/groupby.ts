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
import {DataFrame} from './data_frame';
import {Series} from './series';
import {Table} from './table';
import {NullOrder} from './types/enums'
import {TypeMap} from './types/mappings'

export type AggFunc =
  "sum"|"min"|"max"|"argmin"|"argmax"|"mean"|"count"|"nunique"|"var"|"std"|"median"

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

export type GroupByProps<T extends TypeMap> = {
  obj: DataFrame<T>,
  by: (keyof T)[],
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

export type Groups = {
  keys: DataFrame,
  offsets: number[],
  values?: DataFrame,
}

interface GroupbyConstructor {
  readonly prototype: CudfGroupBy;
  new(props: CudfGroupByProps): CudfGroupBy;
}

interface CudfGroupBy {
  _by: string[];
  _values: DataFrame;
  _getGroups(values?: Table, memoryResource?: MemoryResource): any;
  _agg(func: AggFunc, values: Table, memoryResource?: MemoryResource):
    {keys: Table, cols: [Column]};
}

export class GroupBy<T extends TypeMap> extends(<GroupbyConstructor>CUDF.GroupBy) {
  constructor(props: GroupByProps<T>) {
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
    this._by     = props.by as string[];
    this._values = props.obj.drop(props.by);
  }

  /* Return the Groups for this GroupBy
   *
   *
   */
  getGroups(memoryResource?: MemoryResource): Groups {
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

  /* Apply an aggregation to the group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result'sd
   *   device memory.
   */
  agg(func: AggFunc, memoryResource?: MemoryResource): DataFrame {
    const {keys, cols} = this._agg(func, this._values.asTable(), memoryResource);

    const series_map = {} as any;
    this._by.forEach(
      (name, index) => { series_map[name] = Series.new(keys.getColumnByIndex(index)); });
    this._values.names.forEach((name, index) => { series_map[name] = Series.new(cols[index]); });
    return new DataFrame(series_map);
  }

  /* Compute the sum of values in each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result'sd
   *   device memory.
   */
  sum(memoryResource?: MemoryResource): DataFrame { return this.agg("sum", memoryResource); }

  /* Compute the minimum value in each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result'sd
   *   device memory.
   */
  min(memoryResource?: MemoryResource): DataFrame { return this.agg("min", memoryResource); }

  /* Compute the maximum value in each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result'sd
   *   device memory.
   */
  max(memoryResource?: MemoryResource): DataFrame { return this.agg("max", memoryResource); }

  /* Compute the index of the minimum value in each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result'sd
   *   device memory.
   */
  argmin(memoryResource?: MemoryResource): DataFrame { return this.agg("argmin", memoryResource); }

  /* Compute the index of the maximum value in each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result'sd
   *   device memory.
   */
  argmax(memoryResource?: MemoryResource): DataFrame { return this.agg("argmax", memoryResource); }

  /* Compute the average value each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result'sd
   *   device memory.
   */
  mean(memoryResource?: MemoryResource): DataFrame { return this.agg("mean", memoryResource); }

  /* Compute the size of each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result'sd
   *   device memory.
   */
  count(memoryResource?: MemoryResource): DataFrame { return this.agg("count", memoryResource); }

  /* Compute the number of unique values in each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result'sd
   *   device memory.
   */
  nunique(memoryResource?: MemoryResource): DataFrame {
    return this.agg("nunique", memoryResource);
  }

  /* Compute the variance for each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result'sd
   *   device memory.
   */
  var(memoryResource?: MemoryResource): DataFrame { return this.agg("var", memoryResource); }

  /* Compute the standard deviation for each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result'sd
   *   device memory.
   */
  std(memoryResource?: MemoryResource): DataFrame { return this.agg("std", memoryResource); }

  /* Compute the median value in each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result'sd
   *   device memory.
   */
  median(memoryResource?: MemoryResource): DataFrame { return this.agg("median", memoryResource); }
}
