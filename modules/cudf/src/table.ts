// Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
import {ReadCSVOptions} from './types/csv';
import {TableWriteCSVOptions} from './types/csv';
import {Bool8, DataType, IndexType, Int32} from './types/dtypes';
import {DuplicateKeepOption, NullOrder} from './types/enums';
import {TypeMap} from './types/mappings';
import {ReadORCOptions, TableWriteORCOptions} from './types/orc';
import {ReadParquetOptions, TableWriteParquetOptions} from './types/parquet';

export type ToArrowMetadata = [string | number, ToArrowMetadata[]?];

interface TableConstructor {
  readonly prototype: Table;
  new(props: {columns?: ReadonlyArray<Column>|null}): Table;

  /**
   * Reads a CSV dataset into a set of columns.
   *
   * @param options Settings for controlling reading behavior.
   * @return The CSV data as a Table and a list of column names.
   */
  readCSV<T extends TypeMap = any>(options: ReadCSVOptions<T>):
    {names: (string&keyof T)[], table: Table};

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
   * @summary Explicitly free the device memory associated with this Table.
   */
  dispose(): void;

  /**
   * @summary Return sub-selection from a Table.
   *
   * @description Gathers the rows of the source columns according to `selection`, such that row "i"
   * in the resulting Table's columns will contain row `selection[i]` from the source columns. The
   * number of rows in the result table will be equal to the number of elements in selection. A
   * negative value i in the selection is interpreted as i+n, where `n` is the number of rows in
   * the source table.
   *
   * For dictionary columns, the keys column component is copied and not trimmed if the gather
   * results in abandoned key elements.
   *
   * @param selection A Series of 8/16/32-bit signed or unsigned integer indices to gather.
   * @param nullify_out_of_bounds If `true`, coerce rows that corresponds to out-of-bounds indices
   *   in the selection to null. If `false`, skips all bounds checking for selection values. Pass
   *   false if you are certain that the selection contains only valid indices for better
   *   performance. If `false` and there are out-of-bounds indices in the selection, the behavior
   *   is undefined. Defaults to `false`.
   * @param memoryResource An optional MemoryResource used to allocate the result's device memory.
   */
  gather(selection: Column<IndexType>,
         nullify_out_of_bounds: boolean,
         memoryResource?: MemoryResource): Table;

  /**
   * Return sub-selection from a Table.
   *
   * @param selection A Column of booleans. Rows at true indices are returned, false are omitted.
   * @param memoryResource An optional MemoryResource used to allocate the result's device memory.
   */
  applyBooleanMask(selection: Column<Bool8>, memoryResource?: MemoryResource): Table;

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
  scatterScalar<T extends Scalar[]>(source: T,
                                    indices: Column<IndexType>,
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
               indices: Column<IndexType>,
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
   * Write a Table to Apache ORC file format.
   * @param filePath File path or root directory path.
   * @param options Options controlling ORC writing behavior.
   */
  writeORC(filePath: string, options: TableWriteORCOptions): void;

  /**
   * Write a Table to Apache Parquet file format.
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
   * @summary Explodes a list column's elements.
   *
   * Any list is exploded, which means the elements of the list in each row are expanded into new
   * rows in the output. The corresponding rows for other columns in the input are duplicated.
   *
   * Example:
   * ```
   * [[5,10,15], 100],
   * [[20,25],   200],
   * [[30],      300],
   * returns
   * [5,         100],
   * [10,        100],
   * [15,        100],
   * [20,        200],
   * [25,        200],
   * [30,        300],
   * ```
   *
   * Nulls and empty lists propagate in different ways depending on what is null or empty.
   * ```
   * [[5,null,15], 100],
   * [null,        200],
   * [[],          300],
   * returns
   * [5,           100],
   * [null,        100],
   * [15,          100],
   * ```
   *
   * @note null lists are not included in the resulting table, but nulls inside lists and empty
   * lists will be represented with a null entry for that column in that row.
   *
   * @param {number} index Column index to explode inside the table.
   * @param memoryResource An optional MemoryResource used to allocate the result's device memory.
   */
  explode(index: number, memoryResource?: MemoryResource): Table;

  /**
   * @summary Explodes a list column's elements and includes a position column.
   *
   * Any list is exploded, which means the elements of the list in each row are expanded into new
   * rows in the output. The corresponding rows for other columns in the input are duplicated. A
   * position column is added that has the index inside the original list for each row.
   *
   * Example:
   * ```
   * [[5,10,15], 100],
   * [[20,25],   200],
   * [[30],      300],
   * returns
   * [0,   5,     100],
   * [1,   10,    100],
   * [2,   15,    100],
   * [0,   20,    200],
   * [1,   25,    200],
   * [0,   30,    300],
   * ```
   *
   * Nulls and empty lists propagate in different ways depending on what is null or empty.
   * ```
   * [[5,null,15], 100],
   * [null,        200],
   * [[],          300],
   * returns
   * [0,     5,    100],
   * [1,  null,    100],
   * [2,    15,    100],
   * ```
   *
   * @note null lists are not included in the resulting table, but nulls inside lists and empty
   * lists will be represented with a null entry for that column in that row.
   *
   * @param {number} index Column index to explode inside the table.
   * @param memoryResource An optional MemoryResource used to allocate the result's device memory.
   */
  explodePosition(index: number, memoryResource?: MemoryResource): Table;

  /**
   * @summary Explodes a list column's elements retaining any null entries or empty lists inside.
   *
   * Any list is exploded, which means the elements of the list in each row are expanded into new
   * rows in the output. The corresponding rows for other columns in the input are duplicated.
   *
   * Example:
   * ```
   * [[5,10,15], 100],
   * [[20,25],   200],
   * [[30],      300],
   * returns
   * [5,         100],
   * [10,        100],
   * [15,        100],
   * [20,        200],
   * [25,        200],
   * [30,        300],
   * ```
   *
   * Nulls and empty lists propagate as null entries in the result.
   * ```
   * [[5,null,15], 100],
   * [null,        200],
   * [[],          300],
   * returns
   * [5,           100],
   * [null,        100],
   * [15,          100],
   * [null,        200],
   * [null,        300],
   * ```
   *
   * @param {number} index Column index to explode inside the table.
   * @param memoryResource An optional MemoryResource used to allocate the result's device memory.
   */
  explodeOuter(index: number, memoryResource?: MemoryResource): Table;

  /**
   * @summary Explodes a list column's elements retaining any null entries or empty lists and
   * includes a position column.
   *
   * Any list is exploded, which means the elements of the list in each row are expanded into new
   * rows in the output. The corresponding rows for other columns in the input are duplicated. A
   * position column is added that has the index inside the original list for each row.
   *
   * Example:
   * ```
   * [[5,10,15], 100],
   * [[20,25],   200],
   * [[30],      300],
   * returns
   * [0,   5,    100],
   * [1,  10,    100],
   * [2,  15,    100],
   * [0,  20,    200],
   * [1,  25,    200],
   * [0,  30,    300],
   * ```
   *
   * Nulls and empty lists propagate as null entries in the result.
   * ```
   * [[5,null,15], 100],
   * [null,        200],
   * [[],          300],
   * returns
   * [0,     5,    100],
   * [1,  null,    100],
   * [2,    15,    100],
   * [0,  null,    200],
   * [0,  null,    300],
   * ```
   *
   * @param {number} index Column index to explode inside the table.
   * @param memoryResource An optional MemoryResource used to allocate the result's device memory.
   */
  explodeOuterPosition(index: number, memoryResource?: MemoryResource): Table;

  /**
   * Interleave Series columns of a table into a single column.
   * Converts the column major table `cols` into a row major column.
   * @param memoryResource An optional MemoryResource used to allocate the result's device memory.
   */
  interleaveColumns(memoryResource?: MemoryResource): Column;
}

// eslint-disable-next-line @typescript-eslint/no-redeclare
export const Table: TableConstructor = CUDF.Table;
