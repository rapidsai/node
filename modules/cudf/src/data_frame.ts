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

import {Column, Series, Table} from '@nvidia/cudf';

import {ColumnAccessor} from './column_accessor'
import {ColumnsMap, NullOrder, TypeMap} from './types'

type SeriesMap<T extends TypeMap> = {
  [P in keyof T]: Series<T[P]>
};

export type OrderSpec = {
  ascending: boolean,
  null_order: NullOrder
};

function _seriesToColumns<T extends TypeMap>(data: SeriesMap<T>) {
  const columns = {} as any;
  for (const [name, series] of Object.entries(data)) { columns[name] = series._data; }
  return <ColumnsMap<T>>columns;
}

/**
 * A GPU Dataframe object.
 */
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

  /**
   * The number of rows in each column of this DataFrame
   */
  get numRows() { return this._accessor.columns[0].length; }

  /**
   * The number of columns in this DataFrame
   */
  get numColumns() { return this._accessor.length; }

  /**
   * The names of columns in this DataFrame
   */
  get names() { return this._accessor.names; }

  /**
   * Return a new DataFrame containing only specified columns.
   *
   * @param columns Names of columns keep.
   */
  select<R extends keyof T>(names: R[]) {
    return new DataFrame(this._accessor.selectByColumnNames(names));
  }

  /**
   * Return a new DataFrame with new columns added.
   *
   * @param data mapping of names to new columns to add.
   */
  assign<R extends TypeMap>(data: SeriesMap<R>) {
    return new DataFrame(this._accessor.addColumns(_seriesToColumns(data)));
  }

  /**
   * Return a new DataFrame with specified columns removed.
   *
   * @param names Names of the columns to drop.
   */
  drop<R extends keyof T>(names: R[]) { return new DataFrame(this._accessor.dropColumns(names)); }

  /**
   * Return a series by name.
   *
   * @param name Name of the Series to return.
   */
  get<P extends keyof T>(name: P) { return new Series(this._accessor.get(name)); }

  /**
   * Generate an ordering that sorts DataFrame columns in a specified way
   *
   * @param options mapping of column names to sort order specifications
   *
   * @returns Series containting the permutation indices for the desired sort order
   */
  orderBy<R extends keyof T>(options: {[P in R]: OrderSpec}) {
    const column_orders = new Array<boolean>();
    const null_orders   = new Array<NullOrder>();
    const columns       = new Array<Column<T[keyof T]>>();
    const entries       = Object.entries(options) as [R, OrderSpec][];
    entries.forEach(([name, {ascending, null_order}]) => {
      const col = this.get(name);
      if (col) {
        columns.push(col._data);
        column_orders.push(ascending);
        null_orders.push(null_order);
      }
    });
    // Compute the sorted sorted_indices
    const sorted_indices = new Table({columns}).orderBy(column_orders, null_orders);
    return new Series(sorted_indices);
  }
}
