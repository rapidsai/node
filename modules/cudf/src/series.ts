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

import {MemoryData} from '@nvidia/cuda';
import {DeviceBuffer} from '@nvidia/rmm';
import {Table as ArrowTable} from 'apache-arrow';
import {RecordBatchReader} from 'apache-arrow';
import {VectorType} from 'apache-arrow/interfaces';

import {Column, ColumnProps} from './column';
import {DataFrame} from './data_frame';
import {Table} from './table';
import {Bool8, CUDFToArrowType, DataType, Integral, NullOrder} from './types';

export interface Series {
  getChild(index: number): Series;
  setNullCount(nullCount: number): void;
  setNullMask(mask: DeviceBuffer, nullCount?: number): void;
}

export type SeriesProps<T extends DataType = any> = {
  type: T,
  data?: DeviceBuffer|MemoryData|null,
  offset?: number,
  length?: number,
  nullCount?: number,
  nullMask?: DeviceBuffer|MemoryData|null,
  children?: ReadonlyArray<Series>|null
};

/**
 * One-dimensional GPU array
 */
export class Series<T extends DataType = any> {
  /*private*/ _data: Column<T>;

  constructor(value: SeriesProps<T>|Column<T>) {
    if (value instanceof Column) {
      this._data = value;
    } else {
      const props: ColumnProps = {
        type: value.type.id,
        data: value.data,
        offset: value.offset,
        length: value.length,
        nullCount: value.nullCount,
        nullMask: value.nullMask,
      };
      if (value.children != null) {
        props.children = value.children.map((item: Series) => item._data);
      }
      this._data = new Column<T>(props);
    }
  }

  /**
   * The data type of elements in the underlying data.
   */
  get type() { return this._data.type; }

  /**
   * The GPU buffer for the null-mask
   */
  get mask() { return this._data.mask; }

  /**
   * The number of elements in the underlying data.
   */
  get length() { return this._data.length; }

  /**
   * Whether a null-mask is needed
   */
  get nullable() { return this._data.nullable; }

  /**
   * Whether the Series contains null values.
   */
  get hasNulls() { return this._data.hasNulls; }

  /**
   * Number of null values
   */
  get nullCount() { return this._data.nullCount; }

  /**
   * The number of child columns
   */
  get numChildren() { return this._data.numChildren; }

  /**
   * Return a seb-selection of the Series from the specified indices
   *
   * @param selection
   */
  gather<R extends Integral>(selection: Series<R>) {
    return new Series<T>(this._data.gather(selection._data));
  }

  /**
   * Return a seb-selection of the Series from the specified boolean mask
   *
   * @param mask
   */
  filter(mask: Series<Bool8>) { return new Series<T>(this._data.gather(mask._data)); }

  /**
   * Return a child at the specified index to host memory
   *
   * @param index
   */
  getChild(index: number) { return new Series(this._data.getChild(index)); }

  /**
   * Return a value at the specified index to host memory
   *
   * @param index
   */
  getValue(index: number) { return this._data.getValue(index); }

  // setValue(index: number, value?: this[0] | null);

  /**
   *
   * @param mask The null-mask. Valid values are marked as 1; otherwise 0. The
   * mask bit given the data index idx is computed as:
   * ```
   * (mask[idx // 8] >> (idx % 8)) & 1
   * ```
   * @param nullCount The number of null values. If None, it is calculated
   * automatically.
   */
  setNullMask(mask: DeviceBuffer, nullCount?: number) { this._data.setNullMask(mask, nullCount); }

  /**
   * Convert a column to an Arrow vector in host memory
   */
  toArrow(): VectorType<CUDFToArrowType<T>> {
    const reader = RecordBatchReader.from(new Table({columns: [this._data]}).toArrow([[0]]));
    const column = new ArrowTable(reader.schema, [...reader]).getColumnAt<CUDFToArrowType<T>>(0);
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    return column!.chunks[0] as VectorType<CUDFToArrowType<T>>;
  }

  /**
   * Generate an ordering that sorts the Series in a specified way
   *
   * @param ascending whether to sort ascending (true) or descending (false)
   * @param null_order whether nulls should sort before or after other values
   *
   * @returns Series containting the permutation indices for the desired sort order
   */
  orderBy(ascending = true, null_order: NullOrder = NullOrder.BEFORE) {
    return new DataFrame({"col": this}).orderBy({"col": {ascending, null_order}});
  }

  /**
   * Generate a new Series that is sorted in a specified way
   *
   * @param ascending whether to sort ascending (true) or descending (false)
   *   Default: true
   * @param null_order whether nulls should sort before or after other values
   *   Default: BEFORE
   *
   * @returns Sorted values
   */
  sortValues(ascending = true, null_order: NullOrder = NullOrder.BEFORE) {
    return this.gather(this.orderBy(ascending, null_order));
  }
}
