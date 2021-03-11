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

import {Column} from '../column';
import {DataFrame, SeriesMap} from '../data_frame';
import {Table} from '../table';
import {DataType} from '../types/dtypes';
import {Interpolation, TypeMap} from '../types/mappings';

import {GroupByBase, GroupByBaseProps} from './base';

type TypeMapOf<T extends string, R extends DataType> = {
  [P in T]: R
};

export type GroupByMultipleProps<T extends TypeMap, R extends keyof T, IndexKey extends string> = {
  by: R[],
  index_key: IndexKey,
}&GroupByBaseProps;

export class GroupByMultiple<T extends TypeMap, R extends keyof T, IndexKey extends string> extends
  GroupByBase<T, R> {
  private index_key: IndexKey;
  constructor(obj: DataFrame<T>, props: GroupByMultipleProps<T, R, IndexKey>) {
    super(props, props.by, obj);
    this.index_key = props.index_key;
  }

  protected prepare_results(results: {keys: Table, cols: Column[]}) {
    const {keys, cols} = results;

    type Subset        = Pick<T, R>;
    type Index         = Struct<Subset>;
    type IndexTypeMap  = TypeMapOf<IndexKey, Index>;
    type RestTypeMap   = Omit<T, R>;
    type RestSeriesMap = SeriesMap<RestTypeMap>;

    const rest_map = this._values.names.reduce(
      (xs, key, index) => ({...xs, [key]: Series.new(cols[index])}), {} as RestSeriesMap);

    if (this.index_key in rest_map) {
      throw new Error(`Groupby column name ${this.index_key} already exists`);
    }

    const fields   = [];
    const children = [];
    for (const [index, name] of this._by.entries()) {
      const child = keys.getColumnByIndex(index)
      fields.push(Field.new({name: name as string, type: child.type}));
      const series = Series.new(child);
      children.push(series);
    }

    const index = Series.new<Index>({type: new Struct(fields), children: children});

    return new DataFrame({[this.index_key]: index, ...rest_map} as
                         SeriesMap<IndexTypeMap&RestTypeMap>);
  }

  /**
   * Compute the index of the maximum value in each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  argmax(memoryResource?: MemoryResource) {
    return this.prepare_results(this._cudf_groupby._argmax(this._values.asTable(), memoryResource));
  }

  /**
   * Compute the index of the minimum value in each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  argmin(memoryResource?: MemoryResource) {
    return this.prepare_results(this._cudf_groupby._argmin(this._values.asTable(), memoryResource));
  }

  /**
   * Compute the size of each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  count(memoryResource?: MemoryResource) {
    return this.prepare_results(this._cudf_groupby._count(this._values.asTable(), memoryResource));
  }

  /**
   * Compute the maximum value in each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  max(memoryResource?: MemoryResource) {
    return this.prepare_results(this._cudf_groupby._max(this._values.asTable(), memoryResource));
  }

  /**
   * Compute the average value each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  mean(memoryResource?: MemoryResource) {
    return this.prepare_results(this._cudf_groupby._mean(this._values.asTable(), memoryResource));
  }

  /**
   * Compute the median value in each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  median(memoryResource?: MemoryResource) {
    return this.prepare_results(this._cudf_groupby._median(this._values.asTable(), memoryResource));
  }

  /**
   * Compute the minimum value in each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  min(memoryResource?: MemoryResource) {
    return this.prepare_results(this._cudf_groupby._min(this._values.asTable(), memoryResource));
  }

  /**
   * Return the nth value from each group
   *
   * @param n the index of the element to return
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  nth(n: number, memoryResource?: MemoryResource) {
    return this.prepare_results(this._cudf_groupby._nth(n, this._values.asTable(), memoryResource));
  }

  /**
   * Compute the number of unique values in each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  nunique(memoryResource?: MemoryResource) {
    return this.prepare_results(
      this._cudf_groupby._nunique(this._values.asTable(), memoryResource));
  }

  /**
   * Compute the standard deviation for each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  std(memoryResource?: MemoryResource) {
    return this.prepare_results(this._cudf_groupby._std(this._values.asTable(), memoryResource));
  }

  /**
   * Compute the sum of values in each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  sum(memoryResource?: MemoryResource) {
    return this.prepare_results(this._cudf_groupby._sum(this._values.asTable(), memoryResource));
  }

  /**
   * Compute the variance for each group
   *
   * @param memoryResource The optional MemoryResource used to allocate the result's
   *   device memory.
   */
  var(memoryResource?: MemoryResource) {
    return this.prepare_results(this._cudf_groupby._var(this._values.asTable(), memoryResource));
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
    return this.prepare_results(this._cudf_groupby._quantile(
      q, this._values.asTable(), Interpolation[interpolation], memoryResource));
  }
}
