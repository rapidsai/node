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
import {Column} from '@nvidia/cudf';
import {DeviceBuffer} from '@nvidia/rmm';

import {ColumnProps} from './column'
import {DataType} from './types';

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

  get type() { return this._data.type; }
  get mask() { return this._data.mask; }
  get length() { return this._data.length; }
  get nullable() { return this._data.nullable; }
  get hasNulls() { return this._data.hasNulls; }
  get nullCount() { return this._data.nullCount; }
  get numChildren() { return this._data.numChildren; }

  getChild(index: number) { return new Series(this._data.getChild(index)); }

  getValue(index: number) { return this._data.getValue(index); }
  // setValue(index: number, value?: this[0] | null);

  setNullCount(nullCount: number) { this._data.setNullCount(nullCount); }

  setNullMask(mask: DeviceBuffer, nullCount?: number) { this._data.setNullMask(mask, nullCount); }
}
