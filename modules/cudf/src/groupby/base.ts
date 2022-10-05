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

import {MemoryResource} from '@rapidsai/rmm';

import * as CUDF from '../addon';
import {Column} from '../column';
import {DataFrame, SeriesMap} from '../data_frame';
import {GroupByBaseProps, GroupByProps} from '../groupby';
import {Series} from '../series';
import {Int32, List} from '../types/dtypes';
import {TypeMap} from '../types/mappings';

export type Groups<KeysMap extends TypeMap, ValuesMap extends TypeMap> = {
  keys: DataFrame<KeysMap>,
  offsets: Int32Array,
  values?: DataFrame<ValuesMap>,
}

export class GroupByBase<T extends TypeMap, R extends keyof T> {
  protected _by: R[];
  protected _values: DataFrame<Omit<T, R>>;
  protected _cudf_groupby: InstanceType<typeof CUDF.GroupBy>;

  constructor(props: GroupByBaseProps, by: R[], obj: DataFrame<T>) {
    const table = obj.select(by).asTable();

    const {
      include_nulls   = false,
      keys_are_sorted = false,
      column_order    = [],
      null_precedence = []
    } = props;

    const cudf_props: GroupByProps = {
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

  protected _propagateListFieldNames(name: string&keyof Omit<T, R>, col: Column<List>) {
    (col.type.children[0] as any).name = '0';
    return Series
      .new({
        type: col.type,
        length: col.length,
        nullMask: col.mask,
        nullCount: col.nullCount,
        children: [
          col.getChild<Int32>(0),
          (this._values.get(name) as any).__construct(col.getChild(1))._col
        ]
      })
      ._col;
  }
}
