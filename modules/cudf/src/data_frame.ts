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
import {ColumnNames, TypeMap} from './types'

type SeriesMap<T extends TypeMap> = {
  [P in ColumnNames<T>]: Series<T[P]>
};

export interface DataFrame<T extends TypeMap = any> {
  select<R extends ColumnNames<T>>(columns: R[]): DataFrame<{[P in R]: T[P]}>;
  assign<R extends TypeMap>(data: SeriesMap<R>): DataFrame<T|R>;
  drop<R extends ColumnNames<T>>(names: R[]): DataFrame<Exclude<T, R>>;
  get<P extends ColumnNames<T>>(name: P): Series<T[P]>;
}

function _seriesToColumns<T extends TypeMap>(data: SeriesMap<T>) {
  const columns = {} as any;
  for (const entry of Object.entries(data)) { columns[entry[0]] = entry[1]._data; }
  return columns;
}

export class DataFrame<T extends TypeMap = any> {
  private _table: Table;
  private _accessor: ColumnAccessor<T>;

  constructor(data: ColumnAccessor<T>|SeriesMap<T>) {
    if (data instanceof ColumnAccessor) {
      this._table    = new Table({columns: data.columns});
      this._accessor = data;
    } else {
      const columns  = _seriesToColumns(data);
      const accessor = new ColumnAccessor(columns);
      this._table    = new Table({columns: accessor.columns});
      this._accessor = accessor;
    }
  }

  get numRows() { return this._table.numRows; }

  get numColumns() { return this._table.numColumns; }

  get columns(): ReadonlyArray<ColumnNames<T>> { return this._accessor.names; }

  select<R extends ColumnNames<T>>(columns: R[]): DataFrame<{[P in R]: T[P]}> {
    const column_accessor = this._accessor.selectByColumnNames(columns);
    return new DataFrame(column_accessor);
  }

  assign<R extends TypeMap>(data: SeriesMap<R>): DataFrame<T|R> {
    const columns  = _seriesToColumns(data);
    const accessor = this._accessor.addColumns(columns);
    return new DataFrame(accessor) as DataFrame<T|R>;
  }

  drop<R extends ColumnNames<T>>(names: R[]): DataFrame<Exclude<T, R>> {
    const accessor = this._accessor.dropColumns(names);
    return new DataFrame(accessor);
  }

  get<P extends ColumnNames<T>>(name: P): Series<T[P]> {
    const index = this._accessor.columnNameToColumnIndex(name)
    if (typeof index !== "undefined") { return new Series(this._table.getColumnByIndex(index)); }
    throw new Error(`Series does not exist in the DataFrame: ${name}`);
  }
}

const proxy = new Proxy({}, {
  get(target: any, p: any, df: DataFrame) {
    if (typeof p == 'string') {
      if (df.columns.includes(p)) { return df.get(p); }
    }
    return Reflect.get(target, p, df);
  }
})

Object.setPrototypeOf(DataFrame.prototype, proxy);
