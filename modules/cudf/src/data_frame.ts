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

import {MemoryResource} from '@nvidia/rmm';
import * as arrow from 'apache-arrow';
import {Readable} from 'stream';

import {Column} from './column';
import {ColumnAccessor} from './column_accessor'
import {AbstractSeries, Float32Series, Float64Series, Series} from './series';
import {Table} from './table';
import {CSVToCUDFType, CSVTypeMap, ReadCSVOptions, WriteCSVOptions} from './types/csv';
import {
  Bool8,
  DataType,
  Integral,
} from './types/dtypes'
import {
  NullOrder,
} from './types/enums';
import {
  ColumnsMap,
  TypeMap,
} from './types/mappings';

type SeriesMap<T extends TypeMap> = {
  [P in keyof T]: AbstractSeries<T[P]>
};

export type OrderSpec = {
  ascending: boolean,
  null_order: NullOrder
};

function _seriesToColumns<T extends TypeMap>(data: SeriesMap<T>) {
  const columns = {} as any;
  for (const [name, series] of Object.entries(data)) { columns[name] = series._col; }
  return <ColumnsMap<T>>columns;
}

/**
 * A GPU Dataframe object.
 */
export class DataFrame<T extends TypeMap = any> {
  public static readCSV<T extends CSVTypeMap = any>(options: ReadCSVOptions<T>) {
    const {names, table} = Table.readCSV(options);
    return new DataFrame(new ColumnAccessor(
      names.reduce((map, name, i) => ({...map, [name]: table.getColumnByIndex(i)}),
                   {} as ColumnsMap<{[P in keyof T]: CSVToCUDFType<T[P]>}>)));
  }

  private _accessor: ColumnAccessor<T>;

  constructor(data: ColumnAccessor<T>|SeriesMap<T>) {
    this._accessor =
      (data instanceof ColumnAccessor) ? data : new ColumnAccessor(_seriesToColumns(data));
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
  get<P extends keyof T>(name: P): Series<T[P]> { return Series.new(this._accessor.get(name)); }

  /**
   * Casts each Series in this DataFrame to a new dtype (similar to `static_cast` in C++).
   *
   * @param dataTypes The map from column names to new dtypes.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns DataFrame of Series cast to the new dtype
   */
  cast<R extends {[P in keyof T]?: DataType}>(dataTypes: R, memoryResource?: MemoryResource) {
    const names = this._accessor.names;
    const types = !(dataTypes instanceof arrow.DataType)
                    ? dataTypes
                    : names.reduce((types, name) => ({...types, [name]: dataTypes}), {} as R);
    return new DataFrame(names.reduce(
      (columns, name) => ({
        ...columns,
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        [name]: name in types ? this.get(name).cast(types[name]!, memoryResource) : this.get(name)
      }),
      {} as SeriesMap<Omit<T, keyof R>&R>));
  }

  /**
   * Casts all the Series in this DataFrame to a new dtype (similar to `static_cast` in C++).
   *
   * @param dataType The new dtype.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns DataFrame of Series cast to the new dtype
   */
  castAll<R extends DataType>(dataType: R, memoryResource?: MemoryResource) {
    return new DataFrame(this._accessor.names.reduce(
      (columns, name) => ({...columns, [name]: this.get(name).cast(dataType, memoryResource)}),
      {} as SeriesMap<{[P in keyof T]: R}>));
  }

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
      const child = this.get(name);
      if (child) {
        columns.push(child._col as Column<T[keyof T]>);
        column_orders.push(ascending);
        null_orders.push(null_order);
      }
    });
    // Compute the sorted sorted_indices
    const sorted_indices = new Table({columns}).orderBy(column_orders, null_orders);
    return Series.new(sorted_indices);
  }

  /**
   * Return sub-selection from a DataFrame from the specified indices
   *
   * @param selection
   */
  gather<R extends Integral>(selection: Series<R>) {
    const temp       = new Table({columns: this._accessor.columns});
    const columns    = temp.gather(selection._col);
    const series_map = {} as SeriesMap<T>;
    this._accessor.names.forEach(
      (name, index) => { series_map[name] = Series.new(columns.getColumnByIndex(index)); });
    return new DataFrame(series_map);
  }

  /**
   * Return sub-selection from a DataFrame from the specified boolean mask
   *
   * @param mask
   */
  filter(mask: Series<Bool8>) {
    const temp       = new Table({columns: this._accessor.columns});
    const columns    = temp.gather(mask._col);
    const series_map = {} as SeriesMap<T>;
    this._accessor.names.forEach(
      (name, index) => { series_map[name] = Series.new(columns.getColumnByIndex(index)); });
    return new DataFrame(series_map);
  }

  /**
   * Serialize this DataFrame to CSV format.
   *
   * @param options Options controlling CSV writing behavior.
   *
   * @returns A node ReadableStream of the CSV data.
   */
  toCSV(options: WriteCSVOptions = {}) {
    const readable = new Readable({encoding: 'utf8'});
    new Table({columns: this._accessor.columns}).writeCSV({
      ...options,
      next(buf) { readable.push(buf); },
      complete() { readable.push(null); },
      columnNames: this.names as string[],
    });
    return readable as AsyncIterable<string>;
  }

  _dropNullsRows(how = "any", subset?: string[], thresh?: number): DataFrame {
    const column_names: string[]   = [];
    const column_indices: number[] = [];
    const subset_                  = (subset == undefined) ? this.names as string[] : subset;
    subset_.forEach((col, idx) => {
      if (this.names.includes(col)) {
        column_names.push(col);
        column_indices.push(idx);
      }
    });

    const table_result = new Table({columns: this._accessor.columns});
    let keep_threshold = column_names.length;
    if (thresh !== undefined) {
      keep_threshold = thresh;
    } else if (how == "all") {
      keep_threshold = 1
    }
    const result = table_result.drop_nulls(column_indices, keep_threshold);
    return new DataFrame(this.names.reduce(
      (map, name, i) => ({...map, [name]: Series.new(result.getColumnByIndex(i))}), {}));
  }

  _dropNaNsRows(how = "any", subset?: string[], thresh?: number): DataFrame {
    const column_names: string[]   = [];
    const column_indices: number[] = [];
    const subset_                  = (subset == undefined) ? this.names as string[] : subset;
    subset_.forEach((col, idx) => {
      if (this.names.includes(col) &&
          (this.get(col) instanceof Float32Series || this.get(col) instanceof Float64Series)) {
        column_names.push(col);
        column_indices.push(idx);
      }
    });

    const table_result = new Table({columns: this._accessor.columns});
    let keep_threshold = column_names.length;
    if (thresh !== undefined) {
      keep_threshold = thresh;
    } else if (how == "all") {
      keep_threshold = 1
    }
    const result = table_result.drop_nans(column_indices, keep_threshold);
    return new DataFrame(this.names.reduce(
      (map, name, i) => ({...map, [name]: Series.new(result.getColumnByIndex(i))}), {}));
  }

  _dropNullsColumns(how = "any", subset?: Series, thresh?: number): DataFrame {
    const column_names: string[] = [];
    const df                     = (subset !== undefined) ? this.gather(subset) : this;

    let thresh_value = thresh;

    if (thresh_value == undefined) {
      if (how == "all") {
        thresh_value = 1;
      } else {
        thresh_value = df.numRows
      }
    }
    this.names.forEach(col => {
      if (thresh_value !== undefined) {
        const no_threshold_valid_count =
          (df.get(col).length - df.get(col).nullCount) < thresh_value;
        if (!no_threshold_valid_count) { column_names.push(col as string); }
      }
    });

    return new DataFrame(this._accessor.selectByColumnNames(column_names));
  }

  _dropNaNsColumns(how = "any", subset?: Series, thresh?: number): DataFrame {
    const column_names: string[] = [];
    const df                     = (subset !== undefined) ? this.gather(subset) : this;

    let thresh_value = thresh;

    if (thresh_value == undefined) {
      if (how == "all") {
        thresh_value = 1;
      } else {
        thresh_value = df.numRows
      }
    }
    this.names.forEach(col => {
      if (thresh_value !== undefined) {
        if (df.get(col) instanceof Float32Series || df.get(col) instanceof Float64Series) {
          const nanCount =
            (df.get(col) as Series).nansToNulls().nullCount - this.get(col).nullCount;
          const threshold_valid_count = (df.get(col).length - nanCount) >= thresh_value;
          if (threshold_valid_count) { column_names.push(col as string); }
        }
      }
    });

    return new DataFrame(this._accessor.selectByColumnNames(column_names));
  }

  dropNulls(axis = 0, how = "any", subset?: string[]|Series, thresh?: number): DataFrame {
    if (axis == 0) {
      if (subset instanceof Series) {
        throw new Error(
          "ValueError: subset => for axis=0, expected a list of column_names as subset or undefined for all columns");
      }
      return this._dropNullsRows(how, subset, thresh);
    } else if (axis == 1) {
      if (subset instanceof Array) {
        throw new Error(
          "ValueError: subset => for axis=1, expected a Series<Integer> with indices to select rows or undefined for all rows");
      }
      return this._dropNullsColumns(how, subset, thresh);
    } else {
      throw new Error("ValueError: axis => invalid axis value");
    }
  }

  dropNaNs<R extends Integral>(axis = 0, how = "any", subset?: string[]|Series<R>, thresh?: number):
    DataFrame {
    if (axis == 0) {
      if (subset instanceof Series) {
        throw new Error(
          "ValueError(subset): for axis=0, expected one of {list of column_names, undefined(all columns)}");
      }
      return this._dropNaNsRows(how, subset, thresh);
    } else if (axis == 1) {
      if (subset instanceof Array) {
        throw new Error(
          "ValueError(subset): for axis=1, expected one of {Series<Integer> with indices to select rows, undefined(all rows)}");
      }
      return this._dropNaNsColumns(how, subset, thresh);
    } else {
      throw new Error("ValueError: axis => invalid axis value");
    }
  }
}
