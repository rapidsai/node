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

import {Series} from '@nvidia/cudf';
import {MemoryResource} from '@nvidia/rmm';

import CUDF from '../addon';
import {Column} from '../column';
import {DataFrame, SeriesMap} from '../data_frame';
import {Table} from '../table';
import {NullOrder} from '../types/enums'
import {TypeMap} from '../types/mappings'

export type GroupByBaseProps = {
  include_nulls?: boolean,
  keys_are_sorted?: boolean,
  column_order?: boolean[],
  null_precedence?: NullOrder[],
}

type CudfGroupByProps = {
  keys: Table,
}&GroupByBaseProps;

export type Groups<KeysMap extends TypeMap, ValuesMap extends TypeMap> = {
  keys: DataFrame<KeysMap>,
  offsets: Int32Array,
  values?: DataFrame<ValuesMap>,
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

export class GroupByBase<T extends TypeMap, R extends keyof T> {
  protected _by: R[];
  protected _values: DataFrame<Omit<T, R>>;
  protected _cudf_groupby: CudfGroupBy;

  constructor(props: GroupByBaseProps, by: R[], obj: DataFrame<T>) {
    const table = obj.select(by).asTable();

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

    this._cudf_groupby = new CUDF.GroupBy(cudf_props);

    this._by     = by;
    this._values = obj.drop(by);
  }

  /**
   * Return the Groups for this GroupBy
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  getGroups(memoryResource?: MemoryResource) {
    const {keys, offsets, values} =
      this._cudf_groupby._getGroups(this._values.asTable(), memoryResource);

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
}
