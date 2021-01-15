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
import {ColumnsMap, TypeMap} from './types'

type SeriesMap<T extends TypeMap> = {
  [P in keyof T]: Series<T[P]>
};

function _seriesToColumns<T extends TypeMap>(data: SeriesMap<T>) {
  const columns = {} as any;
  for (const [name, series] of Object.entries(data)) { columns[name] = series._data; }
  return <ColumnsMap<T>>columns;
}

export class DataFrame<T extends TypeMap = any> {
  private _accessor: ColumnAccessor<T>;

  constructor(data: ColumnAccessor<T>|SeriesMap<T>) {
    if (data instanceof ColumnAccessor) {
      this._accessor = data;
    } else {
      const columns  = _seriesToColumns(data);
      const accessor = new ColumnAccessor(columns);
      this._accessor = accessor;
    }
  }

  get numRows() {
    const table = new Table({columns: this._accessor.columns});
    return table.numRows;
  }

  get numColumns() {
    const table = new Table({columns: this._accessor.columns});
    return table.numColumns;
  }

  get names() { return this._accessor.names; }

  select<R extends keyof T>(columns: R[]) {
    return new DataFrame(this._accessor.selectByColumnNames(columns));
  }

  assign<R extends TypeMap>(data: SeriesMap<R>) {
    return new DataFrame(this._accessor.addColumns(_seriesToColumns(data)));
  }

  drop<R extends keyof T>(names: R[]) { return new DataFrame(this._accessor.dropColumns(names)); }

  get<P extends keyof T>(name: P) { return new Series(this._accessor.get(name)); }
}
