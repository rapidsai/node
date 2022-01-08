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

import {Column, PadSideType} from '../column';
import {Series} from '../series';
import {Table} from '../table';
import {
  Bool8,
  Float32,
  Float64,
  Int16,
  Int32,
  Int64,
  Int8,
  Uint16,
  Uint32,
  Uint64,
  Uint8,
  Utf8String
} from '../types/dtypes';

export type ConcatenateOptions = {
  /** String that should inserted between each string from each row. Default is an empty string. */
  separator?: string,
  /**
     String that should be used in place of any null strings found in any column. Default makes a
     null entry in any  produces a null result for that row.
   */
  nullRepr?: string,
  /**
     If true, then the separator is included for null rows if nullRepr is valid. Default is true.
   */
  separatorOnNulls?: boolean,

  /** Device memory resource used to allocate the returned column's device memory. */
  memoryResource?: MemoryResource
};

/**
 * A Series of utf8-string values in GPU memory.
 */
export class StringSeries extends Series<Utf8String> {
  /* eslint-disable @typescript-eslint/no-unused-vars */
  _castAsString(_memoryResource?: MemoryResource): StringSeries {
    return StringSeries.new(this._col);
  }
  /* eslint-enable @typescript-eslint/no-unused-vars */

  _castAsInt8(memoryResource?: MemoryResource): Series<Int8> {
    return Series.new(this._col.stringsToIntegers(new Int8, memoryResource));
  }
  _castAsInt16(memoryResource?: MemoryResource): Series<Int16> {
    return Series.new(this._col.stringsToIntegers(new Int16, memoryResource));
  }
  _castAsInt32(memoryResource?: MemoryResource): Series<Int32> {
    return Series.new(this._col.stringsToIntegers(new Int32, memoryResource));
  }
  _castAsInt64(memoryResource?: MemoryResource): Series<Int64> {
    return Series.new(this._col.stringsToIntegers(new Int64, memoryResource));
  }
  _castAsUint8(memoryResource?: MemoryResource): Series<Uint8> {
    return Series.new(this._col.stringsToIntegers(new Uint8, memoryResource));
  }
  _castAsUint16(memoryResource?: MemoryResource): Series<Uint16> {
    return Series.new(this._col.stringsToIntegers(new Uint16, memoryResource));
  }
  _castAsUint32(memoryResource?: MemoryResource): Series<Uint32> {
    return Series.new(this._col.stringsToIntegers(new Uint32, memoryResource));
  }
  _castAsUint64(memoryResource?: MemoryResource): Series<Uint64> {
    return Series.new(this._col.stringsToIntegers(new Uint64, memoryResource));
  }
  _castAsFloat32(memoryResource?: MemoryResource): Series<Float32> {
    return Series.new(this._col.stringsToFloats(new Float32, memoryResource));
  }
  _castAsFloat64(memoryResource?: MemoryResource): Series<Float64> {
    return Series.new(this._col.stringsToFloats(new Float64, memoryResource));
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
   * // StringSeries
   * Series.new(["foo", "bar", "test"]).getValue(0) // "foo"
   * Series.new(["foo", "bar", "test"]).getValue(2) // "test"
   * Series.new(["foo", "bar", "test"]).getValue(3) // throws index out of bounds error
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
   * // StringSeries
   * const a = Series.new(["foo", "bar", "test"])
   * a.setValue(2, "test1") // inplace update -> Series(["foo", "bar", "test1"])
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
   * Returns an Int32 series the number of bytes of each string in the Series.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new(['Hello', 'Bye', 'Thanks 😊', null]);
   *
   * a.byteCount() // [5, 3, 11, null]
   * ```
   */
  byteCount(memoryResource?: MemoryResource): Series<Int32> {
    return Series.new(this._col.countBytes(memoryResource));
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
   * Returns an Int32 series the length of each string in the Series.
   *
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new(['dog', '', '\n', null]);
   *
   * a.len() // [3, 0, 1 null]
   * ```
   */
  len(memoryResource?: MemoryResource): Series<Int32> {
    return Series.new(this._col.countCharacters(memoryResource));
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

  /**
   * Add padding to each string using a provided character.
   *
   * If the string is already width or more characters, no padding is performed. No strings are
   * truncated.
   *
   * Null string entries result in null entries in the output column.
   *
   * @param width The minimum number of characters for each string.
   * @param side Where to place the padding characters. Default is pad right (left justify).
   * @param fill_char Single UTF-8 character to use for padding. Default is the space character.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new(['aa','bbb','cccc','ddddd', null]);
   *
   * a.pad(4) // ['aa  ','bbb ','cccc','ddddd', null]
   * ```
   */
  pad(width: number, side: PadSideType = 'right', fill_char = ' ', memoryResource?: MemoryResource):
    Series<Utf8String> {
    return Series.new(this._col.pad(width, side, fill_char, memoryResource));
  }

  /**
   * Add '0' as padding to the left of each string.
   *
   * If the string is already width or more characters, no padding is performed. No strings are
   * truncated.
   *
   * This equivalent to `pad(width, 'left', '0')` but is more optimized for this special case.
   *
   * Null string entries result in null entries in the output column.
   *
   * @param width The minimum number of characters for each string.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new(['1234','-9876','+0.34','-342567', null]);
   *
   * a.zfill(6) // ['001234','0-9876','0+0.34','-342567', null]
   * ```
   */
  zfill(width: number, memoryResource?: MemoryResource): Series<Utf8String> {
    return Series.new(this._col.zfill(width, memoryResource));
  }

  /**
   * Applies a JSONPath(string) where each row in the series is a valid json string. Returns New
   * StringSeries containing the retrieved json object strings
   *
   * @param jsonPath The JSONPath string to be applied to each row of the input column
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   *
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = const lines = Series.new([
   *  {foo: {bar: "baz"}},
   *  {foo: {baz: "bar"}},
   * ].map(JSON.stringify)); // StringSeries ['{"foo":{"bar":"baz"}}', '{"foo":{"baz":"bar"}}']
   *
   * a.getJSONObject("$.foo") // StringSeries ['{"bar":"baz"}', '{"baz":"bar"}']
   * a.getJSONObject("$.foo.bar") // StringSeries ["baz", null]
   *
   * // parse the resulting strings using JSON.parse
   * [...a.getJSONObject("$.foo").map(JSON.parse)] // object [{ bar: 'baz' }, { baz: 'bar' }]
   * ```
   */
  getJSONObject(jsonPath: string, memoryResource?: MemoryResource): StringSeries {
    return Series.new(this._col.getJSONObject(jsonPath, memoryResource));
  }

  /**
   * Row-wise concatenates the given list of strings series and returns a single string series
   * result.
   *
   * @param series List of string series to concatenate.
   * @param opts Options for the concatenation
   * @returns New series with concatenated results.
   *
   * @example
   * ```typescript
   * import {StringSeries} from '@rapidsai/cudf';
   * const s = StringSeries.new(['a', 'b', null])
   * const t = StringSeries.new(['foo', null, 'bar'])
   * [...StringSeries.concatenate([s, t])] // ["afoo", null, null]
   * ```
   */
  public static concatenate(series: StringSeries[],
                            opts: ConcatenateOptions = {}): Series<Utf8String> {
    const columns: Column[] = [];
    for (const s of series) { columns.push(s._col); }
    const separator        = opts.separator ?? '';
    const nullRepr         = opts.nullRepr ?? null;
    const separatorOnNulls = opts.separatorOnNulls ?? true;

    return Series.new(Column.concatenate(
      new Table({columns: columns}), separator, nullRepr, separatorOnNulls, opts.memoryResource));
  }
}
