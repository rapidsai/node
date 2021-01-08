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

export type SeriesProps = {
  type: DataType,
  data?: DeviceBuffer|MemoryData|null,
  offset?: number,
  length?: number,
  nullCount?: number,
  nullMask?: DeviceBuffer|MemoryData|null,
  children?: ReadonlyArray<Series>|null
};

export class Series<T extends DataType = any> {
  type: T;
  [key: number]: T ['valueType'];

  /*private*/ _data: Column;

  constructor(value: SeriesProps|Column) {
    if (value instanceof Column) {
      this.type  = value.type.id;
      this._data = value;
    } else {
      this.type                = value.type.id;
      const props: ColumnProps = {
        type: this.type,
        data: value.data,
        offset: value.offset,
        length: value.length,
        nullCount: value.nullCount,
        nullMask: value.nullMask,
      };
      if (value.children != null) {
        props.children = value.children.map((item: Series) => item._data);
      }
      this._data = new Column(props);
    }
  }

  get mask(): DeviceBuffer { return this._data.mask; }
  get length(): number { return this._data.length; }
  get nullable(): boolean { return this._data.nullable; }
  get hasNulls(): boolean { return this._data.hasNulls; }
  get nullCount(): number { return this._data.nullCount; }
  get numChildren(): number { return this._data.numChildren; }

  getChild(index: number): Series { return new Series(this._data.getChild(index)); }

  getValue(index: number): this [0] { return this._data.getValue(index); }
  // setValue(index: number, value?: this[0] | null): void;

  setNullCount(nullCount: number): void { this._data.setNullCount(nullCount); }

  setNullMask(mask: DeviceBuffer, nullCount?: number): void {
    this._data.setNullMask(mask, nullCount);
  }
}

const proxy = new Proxy({}, {
  get(target: any, p: any, series: any) {
    let i: number = p;
    switch (typeof p) {
      // @ts-ignore
      case 'string':
        if (isNaN(i = +p)) { break; }
      // eslint-disable-next-line no-fallthrough
      case 'number':
        if (i > -1 && i < series.length) { return series.getValue(i); }
        return undefined;
    }
    return Reflect.get(target, p, series);
  }
});

Object.setPrototypeOf(Series.prototype, proxy);
