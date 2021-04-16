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

import {Column} from '../column';
import {DataFrame, SeriesMap} from '../data_frame';
import {Series} from '../series';
import {Table} from '../table';
import {DataType, Int32} from '../types/dtypes';
import {Interpolation, TypeMap} from '../types/mappings';

import {GroupByBase, GroupByBaseProps} from './base';

export type GroupBySingleProps<T extends TypeMap, R extends keyof T> = {
  by: R,
}&GroupByBaseProps;

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
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  argmax(memoryResource?: MemoryResource) {
    return this.prepare_results<{[P in keyof T]: P extends R ? T[P] : Int32}>(
      this._cudf_groupby._argmax(this._values.asTable(), memoryResource));
  }

  /**
   * Compute the index of the minimum value in each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  argmin(memoryResource?: MemoryResource) {
    return this.prepare_results<{[P in keyof T]: P extends R ? T[P] : Int32}>(
      this._cudf_groupby._argmin(this._values.asTable(), memoryResource));
  }

  /**
   * Compute the size of each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  count(memoryResource?: MemoryResource) {
    return this.prepare_results<{[P in keyof T]: P extends R ? T[P] : Int32}>(
      this._cudf_groupby._count(this._values.asTable(), memoryResource));
  }

  /**
   * Compute the maximum value in each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  max(memoryResource?: MemoryResource) {
    return this.prepare_results<T>(this._cudf_groupby._max(this._values.asTable(), memoryResource));
  }

  /**
   * Compute the average value each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  mean(memoryResource?: MemoryResource) {
    return this.prepare_results<T>(
      this._cudf_groupby._mean(this._values.asTable(), memoryResource));
  }

  /**
   * Compute the median value in each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  median(memoryResource?: MemoryResource) {
    return this.prepare_results<T>(
      this._cudf_groupby._median(this._values.asTable(), memoryResource));
  }

  /**
   * Compute the minimum value in each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  min(memoryResource?: MemoryResource) {
    return this.prepare_results<T>(this._cudf_groupby._min(this._values.asTable(), memoryResource));
  }

  /**
   * Return the nth value from each group
   *
   * @param n the index of the element to return
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  nth(n: number, memoryResource?: MemoryResource) {
    return this.prepare_results<T>(
      this._cudf_groupby._nth(n, this._values.asTable(), memoryResource));
  }

  /**
   * Compute the number of unique values in each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  nunique(memoryResource?: MemoryResource) {
    return this.prepare_results<{[P in keyof T]: P extends R ? T[P] : Int32}>(
      this._cudf_groupby._nunique(this._values.asTable(), memoryResource));
  }

  /**
   * Compute the standard deviation for each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  std(memoryResource?: MemoryResource) {
    return this.prepare_results<T>(this._cudf_groupby._std(this._values.asTable(), memoryResource));
  }

  /**
   * Compute the sum of values in each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  sum(memoryResource?: MemoryResource) {
    return this.prepare_results<T>(this._cudf_groupby._sum(this._values.asTable(), memoryResource));
  }

  /**
   * Compute the variance for each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  var(memoryResource?: MemoryResource) {
    return this.prepare_results<T>(this._cudf_groupby._var(this._values.asTable(), memoryResource));
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
    return this.prepare_results<T>(this._cudf_groupby._quantile(
      q, this._values.asTable(), Interpolation[interpolation], memoryResource));
  }
}
