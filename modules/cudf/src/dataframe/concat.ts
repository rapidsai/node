// Copyright (c) 2021, NVIDIA CORPORATION.
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

import {DataFrame, SeriesMap} from '../data_frame';
import {Series} from '../series';
import {Table} from '../table';
import {DataType} from '../types/dtypes';
import {CommonTypes, findCommonType} from '../types/mappings';

// clang-format off
export type ConcatTypeMap<D extends DataFrame, T extends unknown[]> =
  T extends []                      ? D['types'] :
  T extends [DataFrame]             ? CommonTypes<D['types'], T[0]['types']> :
  T extends [DataFrame, ...infer U] ? CommonTypes<D['types'], ConcatTypeMap<T[0], U>> :
                                      D['types'] ;
// clang-format on

export function concat<TFirst extends DataFrame, TRest extends DataFrame[]>(first: TFirst,
                                                                            ...others: TRest) {
  const dfs = [first, ...others] as [TFirst, ...TRest];
  const all_column_names =
    [...new Set(dfs.reduce((names: string[], df) => names.concat(df.names), [])).keys()];

    /**
     * Array<Array<(Column|null)>> -- If a DF has a Column for a given name, it will be in the list
     * otherwise there will be a null in that slot. For example:
     * ```
     * concat(new DataFrame({a, b}), new DataFrame({b, c}))
     *
     * columnsPerDF == [
     *  [dfs[0].get("a"), dfs[0].get("b"),            null],
     *  [           null, dfs[1].get("b"), dfs[1].get("c")]
     * ]
     * ```
     */
    const columns_per_df: any[][] =
      dfs.map((df) => all_column_names.map((name) => df.has(name) ? df.get(name)._col : null));

  const num_of_cols = columns_per_df[0].length;
  const num_of_rows = columns_per_df.length;

  /**
   * Array<[number, DataType]> -- Find the first non null dtype in each column, save the column
   * index and dtype in a tuple.
   * ```
   * [[0, Float64], [1, Int32], [2, Float64]],
   * ```
   */
  const first_non_null_dtype: [number, DataType][] = Array(num_of_cols).fill(null);
  first_non_null_dtype.forEach((_, col_idx) => {
    for (let row_idx = 0; row_idx < num_of_rows; ++row_idx) {
      const col = columns_per_df[row_idx][col_idx];
      if (col !== null) { first_non_null_dtype[col_idx] = [row_idx, col.type]; }
    }
  });

  /**
   * Array<DataType> -- Find the common dtype in each column.
   * ```
   * [Float64, Int32, Float64]
   * ```
   */
  const common_dtypes: DataType[] = first_non_null_dtype.map((tuple) => { return tuple[1]; });
  first_non_null_dtype.forEach((tuple, col_idx) => {
    const first_non_null_dtype_idx = tuple[0];
    const start_idx                = first_non_null_dtype_idx + 1;
    for (let row_idx = start_idx; row_idx < num_of_rows; ++row_idx) {
      const first_non_null_dtype = tuple[1];
      const col                  = columns_per_df[row_idx][col_idx];
      if (col !== null) { common_dtypes[col_idx] = findCommonType(first_non_null_dtype, col.type); }
    }
  });

  /**
   * If any columns are null, create an empty column with type of common dtype
   * Otherwise cast the column using the common dtype (if it isn't already the correct type).
   */
  common_dtypes.forEach((common_dtype, col_idx) => {
    for (let row_idx = 0; row_idx < num_of_rows; ++row_idx) {
      const col = columns_per_df[row_idx][col_idx];
      if (col === null) {
        const empty_column_length =
          columns_per_df[row_idx]?.find((col) => col !== null).length ?? 0;
        columns_per_df[row_idx][col_idx] =
          Series.new({type: common_dtype, data: new Array(empty_column_length)})._col;
      } else {
        if (!col.type.compareTo(common_dtype)) {
          columns_per_df[row_idx][col_idx] = col.cast(common_dtype);
        }
      }
    }
  });

  const concatenatedTable = Table.concat(columns_per_df.map((columns) => new Table({columns})));

  type TResultTypeMap = ConcatTypeMap<TFirst, TRest>;

  return new DataFrame(all_column_names.reduce(
    (map, name, index) => ({...map, [name]: Series.new(concatenatedTable.getColumnByIndex(index))}),
    {} as SeriesMap<{[P in keyof TResultTypeMap]: TResultTypeMap[P]}>));
}
