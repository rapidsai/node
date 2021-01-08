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

import {Series, Table} from '@nvidia/cudf';
import {ColumnAccessor} from './column_accessor'

export interface DataFrame {
  [index: number]: any;

  getColumnByIndex(index: number): Series;
  select(columns: ReadonlyArray<number>|ReadonlyArray<string>|null): DataFrame;
  slice(start: number|string, end: number|string): DataFrame;
  updateColumns(props: {columns?: ReadonlyArray<Series>|null}): void;
}

export class DataFrame {
  private _table: Table;
  private _accessor: ColumnAccessor;

  constructor(props: {data: ColumnAccessor|{ [key: string]: Series }}) {
    if (props.data instanceof ColumnAccessor) {
      this._table    = new Table({columns: props.data.columns});
      this._accessor = props.data;
    } else {
      const columns = new Map();
      for (const entry of Object.entries(props.data)) { columns.set(entry [0], entry [1]._data) }
      const accessor = new ColumnAccessor(columns);
      this._table    = new Table({columns: accessor.columns});
      this._accessor = accessor;
    }
  }

  get numRows(): number { return this._table.numRows; }
  get numColumns(): number { return this._table.numColumns; }
  get columns(): ReadonlyArray<string> { return this._accessor.names; }

  select(columns: Array<number>|Array<string>): DataFrame {
    const column_indices: Array<number|undefined> =
      (columns as any []).map((value) => { return this.transformInputLabel(value); });

    const column_accessor = this._accessor.selectByColumnIndices(column_indices);
    return new DataFrame({data: column_accessor});
  }

  slice(start: number|string, end: number|string): DataFrame {
    return new DataFrame({
      data: this._accessor.sliceByColumnIndices(this.transformInputLabel(start),
                                                this.transformInputLabel(end))
    });
  }

  updateColumns(props: {columns?: ReadonlyArray<Series>|null}): void {
    if (props.columns == null) { return this._table.updateColumns({}); }
    return this._table.updateColumns({columns: props.columns.map((item: Series) => item._data)});
  }

  addColumn(name: string, column: Series) {
    this._accessor.insertByColumnName(name, column._data);
    this._table.updateColumns({columns: this._accessor.columns});
  }

  getColumnByIndex(index: number): Series {
    if (typeof this.transformInputLabel(index) !== "undefined" && typeof index === "number") {
      return new Series(this._table.getColumnByIndex(index));
    }
    throw new Error(`Column does not exist in the DataFrame: ${index}`);
  }

  getColumnByName(label: string): Series {
    const index = typeof label === "string" ? this.transformInputLabel(label) : undefined;
    if (typeof index !== "undefined") { return this.getColumnByIndex(index); }
    throw new Error(`Column does not exist in the table: ${label}`);
  }

  drop(props: {columns: Array<string>}) {
    props.columns.forEach((value) => { this._accessor.removeByColumnName(value); });
    this._table.updateColumns({columns: this._accessor.columns});
  }

  private transformInputLabel(label: number|string): number|undefined {
    if (typeof label === "string" && this.columns?.includes(label)) {
      return this._accessor.columnNameToColumnIndex(label)
    } else if (typeof label === "number" && label < this.columns?.length) {
      return label;
    }
    return undefined;
  }
}

const proxy = new Proxy({}, {
  get(target: any, p: any, df: DataFrame) {
    if (typeof p == 'string') {
      if (df.columns.includes(p)) { return df.getColumnByName(p); }
    }
    return Reflect.get(target, p, df);
  }
})

Object.setPrototypeOf(DataFrame.prototype, proxy);
