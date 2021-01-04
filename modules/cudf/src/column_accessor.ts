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

interface ColumnAccessorInterface {
  insertByColumnName(name: string, value: Column): void;
  removeByColumnName(name: string): void;

  selectByColumnName(key: string|undefined): ColumnAccessor|undefined;
  sliceByColumnLabels(start: string, end: string): ColumnAccessor|undefined;
  selectByColumnNames(key: Array<string>): ColumnAccessor|undefined;

  selectByColumnIndex(index: number): ColumnAccessor|undefined;
  sliceByColumnIndices(start: number, end: number): ColumnAccessor|undefined;
  selectByColumnIndices(index: Array<number>): ColumnAccessor|undefined;

  columnNameToColumnIndex(label: string): number|undefined;
  columnIndexToColumnName(index: number): string|undefined;
  columnNamesToColumnIndices(label: Array<string>): Array<number>;
}

export class ColumnAccessor implements ColumnAccessorInterface {
  private _data                                   = new Map<string, Column>();
  private _labels_array: string []                = [];
  private _labels_to_indices: Map<string, number> = new Map();

  set data(value: Map<string, Column>) {
    this._data         = value;
    this._labels_array = Array.from(this._data.keys());
    this._labels_array.forEach((val, index) => this._labels_to_indices.set(val, index));
  }

  private addData(name: string, value: Column) {
    this._data.set(name, value);
    this._labels_array.push(name);
    this._labels_to_indices.set(name, this._labels_array.indexOf(name));
  }

  private removeData(name: string) {
    if (this._data.has(name)) {
      this._data.delete(name);
      this._labels_to_indices.delete(name);
      this._labels_array = this._labels_array.filter(x => x !== name);
    }
  }

  constructor(data: Map<string, Column>) { this.data = data; }

  get names(): ReadonlyArray<string> { return this._labels_array; }

  get columns(): ReadonlyArray<Column> { return Array.from(this._data.values()); }

  get length() { return this._data.size; }

  insertByColumnName(name: string, value: Column) { this.addData(name, value); }

  removeByColumnName(name: string) { this.removeData(name); }

  selectByColumnName(key: string|undefined) {
    if (key != undefined && this._data.has(key)) {
      const temp_val = this._data.get(key);
      if (temp_val != undefined) { return new ColumnAccessor(new Map([[key, temp_val]])); }
    }
    return new ColumnAccessor(new Map());
  }

  sliceByColumnLabels(start: string, end: string) {
    return this.sliceByColumnIndices(this.columnNameToColumnIndex(start),
                                     this.columnNameToColumnIndex(end));
  }

  selectByColumnNames(key: Array<string>) {
    const return_map =
      new Map(Array.from(this._data).filter((x) => { return key.includes(x [0]); }))
    return new ColumnAccessor(return_map);
  }

  selectByColumnIndex(index: number) {
    const label = this.columnIndexToColumnName(index);
    return this.selectByColumnName(label);
  }

  sliceByColumnIndices(start: number|undefined, end: number|undefined) {
    const _start: number = (typeof start === "undefined") ? 0 : start;
    const _end           = (typeof end === "undefined") ? this._labels_array.length : end;

    if (_start >= 0) {
      return new ColumnAccessor(new Map(Array.from(this._data).slice(_start, _end + 1)))
    }
    return new ColumnAccessor(new Map());
  }

  selectByColumnIndices(index: Array<number|undefined>) {
    const return_map = new Map(Array.from(this._data).filter((x) => {
      const temp_val = this.columnNameToColumnIndex(x [0]);
      if (temp_val != undefined) { return index.includes(temp_val); }
      return false;
    }))
    return new ColumnAccessor(return_map);
  }

  columnNameToColumnIndex(label: string): number|undefined {
    return this._labels_to_indices.get(label);
  }

  columnIndexToColumnName(index: number): string|undefined { return this._labels_array [index]; }

  columnNamesToColumnIndices(label: Array<string>): Array<number> {
    const return_array: Array<number> = [];
    for (const _label of label) {
      const temp_index = this.columnNameToColumnIndex(_label);
      if (this._data.has(_label) && temp_index != undefined) { return_array.push(temp_index); }
    }

    return return_array;
  }
}
