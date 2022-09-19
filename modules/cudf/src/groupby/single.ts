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

import {Column} from '../column';
import {DataFrame, SeriesMap} from '../data_frame';
import {GroupByBaseProps} from '../groupby';
import {Series} from '../series';
import {Table} from '../table';
import {DataType, Int32, List} from '../types/dtypes';
import {Interpolation, TypeMap} from '../types/mappings';

import {GroupByBase} from './base';

export interface GroupBySingleProps<T extends TypeMap, R extends keyof T> extends GroupByBaseProps {
  by: R;
}

export class GroupBySingle<T extends TypeMap, R extends keyof T> extends GroupByBase<T, R> {
  constructor(obj: DataFrame<T>, props: GroupBySingleProps<T, R>) { super(props, [props.by], obj); }

  protected prepare_results<U extends {[P in keyof T]: DataType}>(results:
                                                                    {keys: Table, cols: Column[]}) {
    const {keys, cols} = results;
    const series_map   = {} as SeriesMap<U>;

    series_map[this._by[0]] = Series.new(keys.getColumnByIndex(0));
    this._values.names.forEach((name, index) => { series_map[name] = Series.new(cols[index]); });

    return new DataFrame(series_map);
  }

  /**
   * Compute the index of the maximum value in each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's device memory.
   */
  argmax(memoryResource?: MemoryResource) {
    return this.prepare_results<{[P in keyof T]: P extends R ? T[P] : Int32}>(
      this._cudf_groupby._argmax(this._values.asTable(), memoryResource));
  }

  /**
   * Compute the index of the minimum value in each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's device memory.
   */
  argmin(memoryResource?: MemoryResource) {
    return this.prepare_results<{[P in keyof T]: P extends R ? T[P] : Int32}>(
      this._cudf_groupby._argmin(this._values.asTable(), memoryResource));
  }

  /**
   * Compute the size of each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's device memory.
   */
  count(memoryResource?: MemoryResource) {
    return this.prepare_results<{[P in keyof T]: P extends R ? T[P] : Int32}>(
      this._cudf_groupby._count(this._values.asTable(), memoryResource));
  }

  /**
   * Compute the maximum value in each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's device memory.
   */
  max(memoryResource?: MemoryResource) {
    return this.prepare_results<T>(this._cudf_groupby._max(this._values.asTable(), memoryResource));
  }

  /**
   * Compute the average value each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's device memory.
   */
  mean(memoryResource?: MemoryResource) {
    return this.prepare_results<T>(
      this._cudf_groupby._mean(this._values.asTable(), memoryResource));
  }

  /**
   * Compute the median value in each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's device memory.
   */
  median(memoryResource?: MemoryResource) {
    return this.prepare_results<T>(
      this._cudf_groupby._median(this._values.asTable(), memoryResource));
  }

  /**
   * Compute the minimum value in each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's device memory.
   */
  min(memoryResource?: MemoryResource) {
    return this.prepare_results<T>(this._cudf_groupby._min(this._values.asTable(), memoryResource));
  }

  /**
   * Return the nth value from each group
   *
   * @param n the index of the element to return
   * @param {boolean} [include_nulls=true] Whether to include/exclude nulls in list elements.
   * @param memoryResource The optional MemoryResource used to allocate the result's device memory.
   */
  nth(n: number, include_nulls = true, memoryResource?: MemoryResource) {
    return this.prepare_results<T>(
      this._cudf_groupby._nth(this._values.asTable(), memoryResource, n, include_nulls));
  }

  /**
   * Compute the number of unique values in each group
   *
   * @param {boolean} [include_nulls=false] Whether to include/exclude nulls in list elements.
   * @param memoryResource The optional MemoryResource used to allocate the result's device memory.
   */
  nunique(include_nulls = false, memoryResource?: MemoryResource) {
    return this.prepare_results<{[P in keyof T]: P extends R ? T[P] : Int32}>(
      this._cudf_groupby._nunique(this._values.asTable(), memoryResource, include_nulls));
  }

  /**
   * Compute the standard deviation for each group
   *
   * @param {number} [ddof=1] Delta Degrees of Freedom. The divisor used in calculations is N -
   *   ddof, where N represents the number of elements in each group.
   * @param memoryResource The optional MemoryResource used to allocate the result's device memory.
   */
  std(ddof = 1, memoryResource?: MemoryResource) {
    return this.prepare_results<T>(
      this._cudf_groupby._std(this._values.asTable(), memoryResource, ddof));
  }

  /**
   * Compute the sum of values in each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's device memory.
   */
  sum(memoryResource?: MemoryResource) {
    return this.prepare_results<T>(this._cudf_groupby._sum(this._values.asTable(), memoryResource));
  }

  /**
   * Compute the variance for each group
   *
   * @param {number} [ddof=1] Delta Degrees of Freedom. The divisor used in calculations is N -
   *   ddof, where N represents the number of elements in each group.
   * @param memoryResource The optional MemoryResource used to allocate the result's device memory.
   */
  var(ddof = 1, memoryResource?: MemoryResource) {
    return this.prepare_results<T>(
      this._cudf_groupby._var(this._values.asTable(), memoryResource, ddof));
  }

  /**
   * Return values at the given quantile.
   *
   * @param q the quantile to compute, 0 <= q <= 1
   * @param interpolation This optional parameter specifies the interpolation method to use,
   *  when the desired quantile lies between two data points i and j.
   * @param memoryResource The optional MemoryResource used to allocate the result's device memory.
   */
  quantile(q                                         = 0.5,
           interpolation: keyof typeof Interpolation = 'linear',
           memoryResource?: MemoryResource) {
    return this.prepare_results<T>(this._cudf_groupby._quantile(
      this._values.asTable(), memoryResource, q, Interpolation[interpolation]));
  }

  /**
   * Returns a list column of all included elements in the group.
   *
   * @param {boolean} [include_nulls=true] Whether to include/exclude nulls in list elements.
   * @param memoryResource The optional MemoryResource used to allocate the result's device memory.
   */
  collectList(include_nulls = true, memoryResource?: MemoryResource) {
    const {keys, cols} =
      this._cudf_groupby._collect_list(this._values.asTable(), memoryResource, include_nulls);
    this._values.names.forEach((name, index) => {  //
      cols[index] = this._propagateListFieldNames(name, cols[index]);
    });
    return this.prepare_results<{[P in keyof T]: P extends R ? T[P] : List<T[P]>}>({keys, cols});
  }

  /**
   * Returns a lists column of all included elements in the group/series. Within each list, the
   * duplicated entries are dropped out such that each entry appears only once.
   *
   * @param {boolean} [include_nulls=true] Whether to include/exclude nulls in list elements.
   * @param {boolean} [nulls_equal=true] Whether null entries within each list should be considered
   *   equal.
   * @param {boolean} [nans_equal=false] Whether `NaN` values in floating point column should be
   *   considered equal.
   * @param memoryResource The optional MemoryResource used to allocate the result's device memory.
   */
  collectSet(include_nulls = true,
             nulls_equal   = true,
             nans_equal    = false,
             memoryResource?: MemoryResource) {
    const {keys, cols} = this._cudf_groupby._collect_set(
      this._values.asTable(), memoryResource, include_nulls, nulls_equal, nans_equal);
    this._values.names.forEach((name, index) => {  //
      cols[index] = this._propagateListFieldNames(name, cols[index]);
    });
    return this.prepare_results<{[P in keyof T]: P extends R ? T[P] : List<T[P]>}>({keys, cols});
  }
}
