// Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

/* eslint-disable @typescript-eslint/no-non-null-assertion */

import {compareTypes} from 'apache-arrow/visitor/typecomparator';

import {Column} from '../column';
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
  const dfs = ([first, ...others] as [TFirst, ...TRest]).filter((df) => df.numColumns > 0);
  const names =
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
  const rows = dfs.map((df) => {                      //
    return names.map((name) => {                      //
      return df.has(name)                             //
             ? df.get(name)._col as Column<DataType>  //
             : null;
    });
  });

  /**
   * Array<DataType> -- Find the common dtype for each column in the final table.
   * ```
   * [Float64, Int32, Float64]
   * ```
   */
  const commonDtypes = names.map((_, colIdx) => {
    return rows.reduce((commonDtype: DataType|null, columns) => {
      const column = columns[colIdx];
      return !column      ? commonDtype  ///< If Column is null, return the latest common dtype
           : !commonDtype ? column.type  ///< If this is the first non-null Column, use its dtype
                          : findCommonType(commonDtype, column.type);  ///< find the common dtype
    }, null)!;
  });

  /**
   * If any columns are null, create an empty column with type of common dtype
   * Otherwise cast the column using the common dtype (if it isn't already the correct type).
   */
  const tables = dfs.map((df, rowIdx) => {
    // Function to create an empty Column
    const makeEmptyColumn = (() => {
      let data: any[]|undefined;
      return (type: DataType) => {
        // Lazily create and reuse the empty data Array
        data = data || new Array(df.numRows).fill(null);
        return Series.new({type, data})._col;
      };
    })();

    // 1. Create empty Columns for any null slots
    // 2. Cast non-null Columns to the common dtype
    const columns = rows[rowIdx].map((column, colIdx) => {
      const commonDtype = commonDtypes[colIdx];
      if (column === null) {                                 // 1.
        return makeEmptyColumn(commonDtype);
      } else if (!compareTypes(column.type, commonDtype)) {  // 2.
        return column.cast(commonDtype);
      }
      return column;
    });

    // Return each DataFrame to concatenate
    return new DataFrame(
      names.reduce((map, name, index) => ({...map, [name]: columns[index]}), {}));
  });

  const constructChild = (tables[0] as any).__constructChild.bind(tables[0]);

  const result = Table.concat(tables.map((df) => df.asTable()));

  type TResultTypeMap = ConcatTypeMap<TFirst, TRest>;

  // clang-format off
  return new DataFrame(
    names.reduce((map, name, index) =>
    ({...map, [name]: constructChild(name, result.getColumnByIndex(index))}),
    {} as SeriesMap<{[P in keyof TResultTypeMap]: TResultTypeMap[P]}>)
  ) as TResultTypeMap[keyof TResultTypeMap] extends never
    ? never
    : DataFrame<{[P in keyof TResultTypeMap]: TResultTypeMap[P]}>;
  // clang-format on
}
