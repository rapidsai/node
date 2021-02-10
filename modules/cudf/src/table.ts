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

import CUDF from './addon';
import {Column} from './column';
import {CSVTypeMap, ReadCSVOptions, WriteCSVOptions} from './types/csv';
import {
  Bool8,
  DataType,
  Int32,
  Integral,
} from './types/dtypes';
import {
  NullOrder,
} from './types/enums';

export type ToArrowMetadata = [string | number, ToArrowMetadata[]?];

interface TableWriteCSVOptions extends WriteCSVOptions {
  /** Callback invoked for each CSV chunk. */
  next: (chunk: Buffer) => void;
  /** Callback invoked when writing is finished. */
  complete: () => void;
  /** Column names to write in the header. */
  columnNames?: string[];
}

interface TableConstructor {
  readonly prototype: Table;
  new(props: {columns?: ReadonlyArray<Column>|null}): Table;

  /**
   * Reads a CSV dataset into a set of columns.
   *
   * @param options Settings for controlling reading behavior.
   * @return The CSV data as a Table and a list of column names.
   */
  readCSV<T extends CSVTypeMap = any>(options: ReadCSVOptions<T>):
    {names: (keyof T)[], table: Table};
}

/**
 * A low-level wrapper for libcudf Table objects
 */
export interface Table {
  /**
   * Number of columns in the table.
   */
  readonly numColumns: number;

  /**
   * Number of rows in each column of the table.
   */
  readonly numRows: number;

  /**
   * Return sub-selection from a Table
   *
   * @param selection
   */
  gather(selection: Column<Integral|Bool8>): Table;

  /**
   * Get the Column at a specified index
   *
   * @param index The index position of the column to return
   *
   * @reurns The Column located at `index`
   */
  getColumnByIndex<T extends DataType = any>(index: number): Column<T>;

  /**
   * Generate an ordering that sorts Table columns in a specified way
   *
   * @param column_orders The desired sort order for each column. Size must
   * be equal to `numColumns`.
   * @param null_orders Indicates how null values compare against all
   * other values in a column
   *
   * @returns Column of permutation indices for the desired sort order
   */
  orderBy(column_orders: boolean[], null_orders: NullOrder[]): Column<Int32>;
  toArrow(names: ToArrowMetadata[]): Uint8Array;

  /**
   * Write this Table to CSV file format.
   * @param options Settings for controlling writing behavior.
   */
  writeCSV(options: TableWriteCSVOptions): void;
}

// eslint-disable-next-line @typescript-eslint/no-redeclare
export const Table: TableConstructor = CUDF.Table;
