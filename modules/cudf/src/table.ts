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

import {MemoryData} from '@rapidsai/cuda';
import {DeviceBuffer, MemoryResource} from '@rapidsai/rmm';

import CUDF from './addon';
import {Column} from './column';
import {Scalar} from './scalar';
import {CSVTypeMap, ReadCSVOptions, WriteCSVOptions} from './types/csv';
import {Bool8, DataType, IndexType, Int32} from './types/dtypes';
import {DuplicateKeepOption, NullOrder} from './types/enums';
import {TypeMap} from './types/mappings';
import {ReadORCOptions} from './types/orc';
import {ReadParquetOptions, WriteParquetOptions} from './types/parquet';

export type ToArrowMetadata = [string | number, ToArrowMetadata[]?];

interface TableWriteCSVOptions extends WriteCSVOptions {
  /** Callback invoked for each CSV chunk. */
  next: (chunk: Buffer) => void;
  /** Callback invoked when writing is finished. */
  complete: () => void;
  /** Column names to write in the header. */
  columnNames?: string[];
}

interface TableWriteParquetOptions extends WriteParquetOptions {
  /** Column names to write in the header. */
  columnNames?: string[];
}

interface TableWriteORCOptions {
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

  /**
   * Reads an ORC dataset into a set of columns.
   *
   * @param options Settings for controlling reading behavior.
   * @return The ORC data as a Table and a list of column names.
   */
  readORC(options: ReadORCOptions): {names: string[], table: Table};

  /**
   * Reads an Apache Parquet dataset into a set of columns.
   *
   * @param options Settings for controlling reading behavior.
   * @return The CSV data as a Table and a list of column names.
   */
  readParquet(options: ReadParquetOptions): {names: string[], table: Table};

  /**
   * Adapts an arrow Table in IPC format into a set of columns.
   *
   * @param memory A buffer holding Arrow table
   * @return The Arrow data as a Table and a list of column names.
   */
  fromArrow(memory: DeviceBuffer|MemoryData): {names: string[], table: Table};

  /**
   * Returns tables concatenated to each other.
   *
   * @param tables The tables to concatenate
   * @param memoryResource An optional MemoryResource used to allocate the result's device memory.
   */
  concat(tables: Table[], memoryResource?: MemoryResource): Table;

  /**
   * Returns a pair of row index vectors corresponding to a full (outer) join between the specified
   * tables.
   *
   * @param left_keys The left table
   * @param right_keys The right table
   * @param nullEquality controls whether null join-key values should match or not
   * @param memoryResource An optional MemoryResource used to allocate the result's device memory.
   */
  fullJoin(left: Table, right: Table, nullEquality: boolean, memoryResource?: MemoryResource):
    [Column<Int32>, Column<Int32>];

  /**
   * Returns a pair of row index vectors corresponding to an inner join between the specified
   * tables.
   *
   * @param left_keys The left table
   * @param right_keys The right table
   * @param nullEquality controls whether null join-key values should match or not
   * @param memoryResource An optional MemoryResource used to allocate the result's device memory.
   */
  innerJoin(left: Table, right: Table, nullEquality: boolean, memoryResource?: MemoryResource):
    [Column<Int32>, Column<Int32>];

  /**
   * Returns a pair of row index vectors corresponding to a left join between the specified tables.
   *
   * @param left_keys The left table
   * @param right_keys The right table
   * @param nullEquality controls whether null join-key values should match or not
   * @param memoryResource An optional MemoryResource used to allocate the result's device memory.
   */
  leftJoin(left: Table, right: Table, nullEquality: boolean, memoryResource?: MemoryResource):
    [Column<Int32>, Column<Int32>];

  /**
   * Returns an index vectors corresponding to a left semijoin between the specified tables.
   *
   * @param left_keys The left table
   * @param right_keys The right table
   * @param nullEquality controls whether null join-key values should match or not
   * @param memoryResource An optional MemoryResource used to allocate the result's device memory.
   */
  leftSemiJoin(left: Table, right: Table, nullEquality: boolean, memoryResource?: MemoryResource):
    Column<Int32>;

  /**
   * Returns an index vectors corresponding to a left antijoin between the specified tables.
   *
   * @param left_keys The left table
   * @param right_keys The right table
   * @param nullEquality controls whether null join-key values should match or not
   * @param memoryResource An optional MemoryResource used to allocate the result's device memory.
   */
  leftAntiJoin(left: Table, right: Table, nullEquality: boolean, memoryResource?: MemoryResource):
    Column<Int32>;
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
  gather(selection: Column<IndexType|Bool8>, nullify_out_of_bounds?: boolean): Table;

  /**
   * Scatters row of values into this Table according to provided indices.
   *
   * @param source A column of values to be scattered in to this Series
   * @param indices A column of integral indices that indicate the rows in the this Series to be
   *   replaced by `value`.
   * @param check_bounds Optionally perform bounds checking on the indices and throw an error if any
   *   of its values are out of bounds (default: false).
   * @param memoryResource An optional MemoryResource used to allocate the result's device memory.
   */
  scatterScalar<T extends TypeMap = any>(source: (Scalar<T[keyof T]>)[],
                                         indices: Column<Int32>,
                                         check_bounds?: boolean,
                                         memoryResource?: MemoryResource): Table;

  /**
   * Scatters a Table of values into this Table according to provided indices.
   *
   * @param value A value to be scattered in to this Series
   * @param indices A column of integral indices that indicate the rows in the this Series to be
   *   replaced by `value`.
   * @param check_bounds Optionally perform bounds checking on the indices and throw an error if any
   *   of its values are out of bounds (default: false).
   * @param memoryResource An optional MemoryResource used to allocate the result's device memory.
   */
  scatterTable(source: Table,
               indices: Column<Int32>,
               check_bounds?: boolean,
               memoryResource?: MemoryResource): Table;

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

  /**
   * Write a Table to ORC file format.
   * @param filePath File path or root directory path.
   * @param options Options controlling ORC writing behavior.
   */
  writeORC(filePath: string, options: TableWriteORCOptions): void;

  /**
   * Write a Table to parquet file format.
   * @param filePath File path or root directory path.
   * @param options Options controlling parquet writing behavior.
   */
  writeParquet(filePath: string, options: TableWriteParquetOptions): void;

  dropNans(keys: number[], threshold: number): Table;
  dropNulls(keys: number[], threshold: number): Table;
  dropDuplicates(keys: number[],
                 keep: DuplicateKeepOption,
                 nullsEqual: boolean,
                 nullsFirst: boolean,
                 memoryResource?: MemoryResource): Table;
  /**
   * Interleave Series columns of a table into a single column.
   * Converts the column major table `cols` into a row major column.
   * @param memoryResource An optional MemoryResource used to allocate the result's device memory.
   */
  interleaveColumns(memoryResource?: MemoryResource): Column;
}

// eslint-disable-next-line @typescript-eslint/no-redeclare
export const Table: TableConstructor = CUDF.Table;
