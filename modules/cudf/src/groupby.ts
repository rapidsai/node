// Copyright (c) 2020, NVIDIA CORPORATION.
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
import {DataFrame} from './data_frame';
import {Series} from './series';
import {Table} from './table';
import {NullOrder, TypeMap} from './types'

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

interface GroupbyConstructor {
  readonly prototype: CudfGroupBy;
  new(props: CudfGroupByProps): CudfGroupBy;
}

interface CudfGroupBy {
  _by: string[];
  _values: DataFrame;
  _getGroups(values?: Table, memoryResource?: MemoryResource): any;
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

  getGroups(memoryResource?: MemoryResource): any {
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
}
