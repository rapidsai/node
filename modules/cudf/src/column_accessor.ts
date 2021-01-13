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

import {Column} from "./column";
import {ColumnNames, TypeMap} from './types';

type ColumnsMap<T extends TypeMap> = {
  [P in keyof T]: Column<T[P]>
};

export class ColumnAccessor<T extends TypeMap = any> {
  private _data: ColumnsMap<T>;
  private _labels_to_indices: Map<ColumnNames<T>, number> = new Map();

  constructor(data: ColumnsMap<T>) {
    this._data = data;
    this.names.forEach((val, index) => this._labels_to_indices.set(val, index));
  }

  get names(): ReadonlyArray<ColumnNames<T>> { return Object.keys(this._data) as ColumnNames<T>[]; }

  get columns(): ReadonlyArray<Column> { return Object.values(this._data); }

  get length() { return this._labels_to_indices.size; }

  get<R extends ColumnNames<T>>(key: R) { return this._data[key]; }

  addColumns<R extends TypeMap>(data: ColumnsMap<R>): ColumnAccessor<T|R> {
    return new ColumnAccessor({...this._data, ...data} as ColumnsMap<T&R>);
  }

  dropColumns<R extends ColumnNames<T>>(names: R[]): ColumnAccessor<Exclude<T, R>> {
    const data     = {} as any;
    const filtered = Object.keys(this._data).filter((x) => { return !names.includes(x as R); });
    for (const name of filtered) { data[name] = this._data[name]; }
    return new ColumnAccessor(data);
  }

  selectByColumnName<R extends ColumnNames<T>>(name: R): ColumnAccessor {
    return this.selectByColumnNames([name]);
  }

  selectByColumnNames<R extends ColumnNames<T>>(names: R[]) {
    const data     = {} as any;
    const filtered = Object.keys(this._data).filter((x) => { return names.includes(x as R); });
    for (const name of filtered) { data[name] = this._data[name]; }
    return new ColumnAccessor(data);
  }

  columnNameToColumnIndex(name: ColumnNames<T>): number|undefined {
    return this._labels_to_indices.get(name);
  }
}
