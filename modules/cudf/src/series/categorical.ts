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

import {MemoryResource} from '@rapidsai/rmm';
import {compareTypes} from 'apache-arrow/visitor/typecomparator';

import {Column} from '../column';
import {DataFrame} from '../data_frame';
import {scope} from '../scope';
import {Series} from '../series';
import {
  Categorical,
  DataType,
  Int16,
  Int32,
  Int64,
  Int8,
  Integral,
  Uint16,
  Uint32,
  Uint64,
  Uint8,
  Utf8String
} from '../types/dtypes';

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
   * @inheritdoc
   */
  public encodeLabels<R extends Integral = Uint32>(
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    _categories: Series<Categorical<T>> = this,
    type: R                             = new Uint32 as R,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    _nullSentinel: R['scalarType'] = -1,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    _memoryResource?: MemoryResource): Series<R> {
    return compareTypes(this.type.indices, type) ? this.codes as Series<R>  //
                                                 : this.codes.cast(type);
  }

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

  _castAsInt8(memoryResource?: MemoryResource): Series<Int8> {
    return this._castCategories(new Int8, memoryResource);
  }
  _castAsInt16(memoryResource?: MemoryResource): Series<Int16> {
    return this._castCategories(new Int16, memoryResource);
  }
  _castAsInt32(memoryResource?: MemoryResource): Series<Int32> {
    return this._castCategories(new Int32, memoryResource);
  }
  _castAsInt64(memoryResource?: MemoryResource): Series<Int64> {
    return this._castCategories(new Int64, memoryResource);
  }
  _castAsUint8(memoryResource?: MemoryResource): Series<Uint8> {
    return this._castCategories(new Uint8, memoryResource);
  }
  _castAsUint16(memoryResource?: MemoryResource): Series<Uint16> {
    return this._castCategories(new Uint16, memoryResource);
  }
  _castAsUint32(memoryResource?: MemoryResource): Series<Uint32> {
    return this._castCategories(new Uint32, memoryResource);
  }
  _castAsUint64(memoryResource?: MemoryResource): Series<Uint64> {
    return this._castCategories(new Uint64, memoryResource);
  }
  _castAsString(memoryResource?: MemoryResource): Series<Utf8String> {
    return this._castCategories(new Utf8String, memoryResource);
  }
  _castAsCategorical<R extends Categorical>(type: R, memoryResource?: MemoryResource): Series<R> {
    return Series.new({
      type,
      length: this.length,
      nullMask: this.mask,
      children: [this.codes, this.categories.cast(type, memoryResource)]
    });
  }

  protected _castCategories<R extends DataType>(type: R,
                                                memoryResource?: MemoryResource): Series<R> {
    const result = this.categories.gather(this.codes).cast(type, memoryResource);
    result.setNullMask(this.mask);
    return result;
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
    return scope(() => {
      const new_cats = scope(() => {
        const uniq = scope(() => {
          return new DataFrame({
                   value: Series.new<T>(<any>categories),
                   order: Series.sequence({size: categories.length})
                 })
            .groupBy({by: 'value'})
            .nth(0);
        });
        return uniq.sortValues({order: {ascending: true}}, memoryResource).get('value');
      });

      const new_codes = scope(() => {
        // 1. Join the old and new categories to align their codes
        const aligned = scope(() => {
          const cur_cats  = this.categories;
          const old_codes = Series.sequence({size: cur_cats.length});
          const new_codes = Series.sequence({size: new_cats.length});
          const old_df    = new DataFrame({old_codes, categories: cur_cats});
          const new_df    = new DataFrame({new_codes, categories: new_cats});

          return old_df.join({how: 'left', on: ['categories'], other: new_df});
        });
        // 2. Join the old and new codes to "recode" the new categories
        const cur_index = Series.sequence({size: this.codes.length});
        const cur_df    = new DataFrame({old_codes: this.codes, cur_index});

        return cur_df.join({how: 'left', on: ['old_codes'], other: aligned})
          .drop(['old_codes'])
          .sortValues({cur_index: {ascending: true}}, memoryResource)
          .get('new_codes');
      }, [new_cats]);

      return this.__construct(new Column({
        type: this.type,
        length: this.length,
        nullMask: this.mask,
        children: [new_codes._col, new_cats._col]
      }));
    }, [this.codes, this.categories, categories]);
  }
}
