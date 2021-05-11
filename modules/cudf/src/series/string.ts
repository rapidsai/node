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
import * as arrow from 'apache-arrow';

import {Series} from '../series';
import {Bool8, DataType, Int32, Uint8, Utf8String} from '../types/dtypes';

/**
 * A Series of utf8-string values in GPU memory.
 */
export class StringSeries extends Series<Utf8String> {
  /**
   * Casts the values to a new dtype (similar to `static_cast` in C++).
   *
   * @param dataType The new dtype.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns Series of same size as the current Series containing result of the `cast` operation.
   */
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  cast<R extends DataType>(dataType: R, _memoryResource?: MemoryResource): Series<R> {
    throw new Error(`Cast from ${arrow.Type[this.type.typeId]} to ${
      arrow.Type[dataType.typeId]} not implemented`);
  }

  /**
   * Return a value at the specified index to host memory
   *
   * @param index the index in this Series to return a value for
   *
   * @example
   * ```typescript
   * import {Series} from "@rapidsai/cudf";
   *
   * // Float64Series
   * Series.new([1, 2, 3]).getValue(0) // 1
   * // StringSeries
   * Series.new(["foo", "bar", "test"]).getValue(2) // "test"
   * // Bool8Series
   * Series.new([false, true, true]).getValue(3) // throws index out of bounds error
   * ```
   */
  getValue(index: number) { return this._col.getValue(index); }

  /**
   * set value at the specified index
   *
   * @param index the index in this Series to set a value for
   * @param value the value to set at `index`
   *
   * @example
   * ```typescript
   * import {Series} from "@rapidsai/cudf";
   *
   * // Float64Series
   * const a = Series.new([1, 2, 3]);
   * a.setValue(0, -1) // inplace update [-1, 2, 3]
   *
   * // StringSeries
   * const b = Series.new(["foo", "bar", "test"])
   * b.setValue(1,"test1") // inplace update ["foo", "test1", "test"]
   * // Bool8Series
   * const c = Series.new([false, true, true])
   * c.cetValue(2, false) // inplace update [false, true, false]
   * ```
   */
  setValue(index: number, value: string): void { this._col = this.scatter(value, [index])._col; }

  /**
   * Series of integer offsets for each string
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new(["foo", "bar"]);
   *
   * a.offsets // Int32Array(3) [ 0, 3, 6 ]
   * ```
   */
  // TODO: Account for this.offset
  get offsets() { return Series.new(this._col.getChild<Int32>(0)); }

  /**
   * Series containing the utf8 characters of each string
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new(["foo", "bar"]);
   *
   * a.data // Uint8Array(6) [ 102, 111, 111, 98, 97, 114 ]
   * ```
   */
  // TODO: Account for this.offset
  get data() { return Series.new(this._col.getChild<Uint8>(1)); }

  /**
   * Returns a boolean series identifying rows which match the given regex pattern.
   *
   * @param pattern Regex pattern to match to each string.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   *
   * The regex pattern strings accepted are described here:
   *
   * https://docs.rapids.ai/api/libcudf/stable/md_regex.html
   *
   * A RegExp may also be passed, however all flags are ignored (only `pattern.source` is used)
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new(['Finland','Colombia','Florida', 'Russia','france']);
   *
   * // items starting with F (only upper case)
   * a.containsRe(/^F/) // [true, false, true, false, false]
   * // items starting with F or f
   * a.containsRe(/^[Ff]/) // [true, false, true, false, true]
   * // items ending with a
   * a.containsRe("a$") // [false, true, true, true, false]
   * // items containing "us"
   * a.containsRe("us") // [false, false, false, true, false]
   * ```
   */
  containsRe(pattern: string|RegExp, memoryResource?: MemoryResource): Series<Bool8> {
    const pat_string = pattern instanceof RegExp ? pattern.source : pattern;
    return Series.new(this._col.containsRe(pat_string, memoryResource));
  }

  /**
   * Returns an Int32 series the number of times the given regex pattern matches
   * in each string.
   *
   * @param pattern Regex pattern to match to each string.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   *
   * The regex pattern strings accepted are described here:
   *
   * https://docs.rapids.ai/api/libcudf/stable/md_regex.html
   *
   * A RegExp may also be passed, however all flags are ignored (only `pattern.source` is used)
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new(['Finland','Colombia','Florida', 'Russia','france']);
   *
   * // count occurences of "o"
   * a.countRe(/o/) // [0, 2, 1, 0, 0]
   * // count occurences of "an"
   * a.countRe('an') // [1, 0, 0, 0, 1]
   *
   * // get number of countries starting with F or f
   * a.countRe(/^[fF]).count() // 3
   * ```
   */
  countRe(pattern: string|RegExp, memoryResource?: MemoryResource): Series<Int32> {
    const pat_string = pattern instanceof RegExp ? pattern.source : pattern;
    return Series.new(this._col.countRe(pat_string, memoryResource));
  }

  /**
   * Returns a boolean series identifying rows which match the given regex pattern
   * only at the beginning of the string
   *
   * @param pattern Regex pattern to match to each string.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   *
   * The regex pattern strings accepted are described here:
   *
   * https://docs.rapids.ai/api/libcudf/stable/md_regex.html
   *
   * A RegExp may also be passed, however all flags are ignored (only `pattern.source` is used)
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new(['Finland','Colombia','Florida', 'Russia','france']);
   *
   * // start of item contains "C"
   * a.matchesRe(/C/) // [false, true, false, false, false]
   * // start of item contains "us", returns false since none of the items start with "us"
   * a.matchesRe('us') // [false, false, false, false, false]
   * ```
   */
  matchesRe(pattern: string|RegExp, memoryResource?: MemoryResource): Series<Bool8> {
    const pat_string = pattern instanceof RegExp ? pattern.source : pattern;
    return Series.new(this._col.matchesRe(pat_string, memoryResource));
  }
}
