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

import {MemoryResource} from '@rapidsai/rmm';

import {Column} from '../column';
import {ColumnAccessor} from '../column_accessor';
import {DataFrame} from '../data_frame';
import {Series} from '../series';
import {Categorical, DataType, Int32} from '../types/dtypes';

/**
 * A Series of dictionary-encoded values in GPU memory.
 */
export class CategoricalSeries<T extends DataType> extends Series<Categorical<T>> {
  /**
   * @summary The Series of codes.
   */
  public get codes() { return Series.new(this._col.getChild<Int32>(0)); }

  /**
   * @summary The Series of categories.
   */
  public get categories() { return Series.new(this._col.getChild<T>(1)); }

  /**
   * @summary Get a value at the specified index to host memory
   *
   * @param index the index in this Series to return a value for
   *
   * @example
   * ```typescript
   * import {Series} from "@rapidsai/cudf";
   *
   * // StringSeries
   * Series.new(["foo", "bar", "test"]).getValue(0) // "foo"
   * Series.new(["foo", "bar", "test"]).getValue(2) // "test"
   * Series.new(["foo", "bar", "test"]).getValue(3) // throws index out of bounds error
   * ```
   */
  public getValue(index: number) { return this._col.getValue(index); }

  /**
   * @summary Set a value at the specified index.
   *
   * @param index the index in this Series to set a value for
   * @param value the value to set at `index`
   *
   * @example
   * ```typescript
   * import {Series} from "@rapidsai/cudf";
   *
   * // StringSeries
   * const a = Series.new(["foo", "bar", "test"])
   * a.setValue(2, "test1") // inplace update -> Series(["foo", "bar", "test1"])
   * ```
   */
  public setValue(index: number, value: T['scalarType']): void {
    this._col = this.scatter(value, [index])._col;
  }

  /**
   * @summary Casts the values to a new dtype (similar to `static_cast` in C++).
   *
   * @param type The new dtype.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns Series of same size as the current Series containing result of the `cast` operation.
   */
  public cast<R extends DataType>(type: R, memoryResource?: MemoryResource): Series<R> {
    if (type instanceof Categorical) {
      return this.castCategories(type.dictionary, memoryResource);
    }
    const result = this.categories.gather(this.codes).cast(type, memoryResource) as Series<R>;
    result.setNullMask(this.mask);
    return result;
  }

  /**
   * @summary Casts the categories to a new dtype (similar to `static_cast` in C++), keeping the
   * current codes.
   *
   * @param type The new categories dtype.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns Series of same size as the current Series containing result of the `cast` operation.
   */
  public castCategories<R extends DataType>(type: R, memoryResource?: MemoryResource): Series<R> {
    return Series.new(new Column({
      type,
      length: this.length,
      nullMask: this.mask,
      children: [this.codes._col, this.categories.cast(type, memoryResource)._col]
    }));
  }

  /**
   * @summary Create a new CategoricalColumn with the categories set to the specified `categories`.
   *
   * @param categories The new categories
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns CategoricalSeries of same size as the current Series with the new categories.
   */
  public setCategories(categories: Series<T>|(T['scalarType'][]),
                       memoryResource?: MemoryResource): Series<Categorical<T>> {
    const cur_cats = this.categories._col as Column<T>;
    // Pass the memoryResource here because the libcudf semantics are that it should only be used to
    // allocate memory for the columns that are returned.
    const new_cats = Series.new<T>(<any>categories).unique(true, memoryResource)._col as Column<T>;

    const cur_codes = this.codes._col;
    const old_codes =
      Series.sequence({type: new Int32, init: 0, step: 1, size: cur_cats.length})._col;
    const new_codes =
      Series.sequence({type: new Int32, init: 0, step: 1, size: new_cats.length})._col;
    const cur_index =
      Series.sequence({type: new Int32, init: 0, step: 1, size: cur_codes.length})._col;

    const old_df = new DataFrame(new ColumnAccessor({old_codes: old_codes, categories: cur_cats}));
    const cur_df = new DataFrame(new ColumnAccessor({old_codes: cur_codes, cur_index: cur_index}));
    const new_df = new DataFrame(new ColumnAccessor({new_codes: new_codes, categories: new_cats}));

    //
    // 1. Join the old and new categories to align their codes
    // 2. Join the old and new codes to "recode" the new categories
    //
    // Note: Written as a single expression so the intermediate memory allocated for the `join` and
    // `sortValues` calls are GC'd as soon as possible.
    //
    const out_codes =
      // 2.
      cur_df
        .join({
          how: 'left',
          on: ['old_codes'],
          // Pass the memoryResource here because the libcudf semantics are that it should only be
          // used to allocate memory for the columns that are returned.
          memoryResource,
          // 1.
          other: old_df.join({
            how: 'left',
            on: ['categories'],
            // Pass the memoryResource here because the libcudf semantics are that it should only be
            // used to allocate memory for the columns that are returned.
            memoryResource,
            other: new_df,
          })
        })
        .sortValues({cur_index: {ascending: true}})
        .get('new_codes')
        ._col;

    return Series.new(new Column({
      type: this.type,
      length: this.length,
      nullMask: this.mask,
      children: [out_codes, new_cats]
    }));
  }
}
