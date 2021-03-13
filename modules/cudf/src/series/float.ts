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

import {Float32Buffer, Float64Buffer} from '@nvidia/cuda';
import {MemoryResource} from '@rapidsai/rmm';

import {Series} from '../series';
import {Float32, Float64, FloatingPoint} from '../types/dtypes';

import {NumericSeries} from './numeric';

/**
 * A base class for Series of 32 or 64-bit floating-point values in GPU memory.
 */
abstract class FloatSeries<T extends FloatingPoint> extends NumericSeries<T> {
  /**
   * Creates a column of `BOOL8` elements indicating the presence of `NaN` values in a
   * column of floating point values. The output element at row `i` is `true` if the element in
   * `input` at row i is `NAN`, else `false`
   *
   * @param memoryResource Memory resource used to allocate the result Series's device memory.
   * @returns A non-nullable column of `BOOL8` elements with `true` representing `NAN`
   *   values
   */
  isNaN(memoryResource?: MemoryResource) { return Series.new(this._col.isNaN(memoryResource)); }

  /**
   * Creates a column of `BOOL8` elements indicating the absence of `NaN` values in a
   * column of floating point values. The output element at row `i` is `false` if the element in
   * `input` at row i is `NAN`, else `true`
   *
   * @param memoryResource Memory resource used to allocate the result Series's device memory.
   * @returns A non-nullable column of `BOOL8` elements with `true` representing `NAN`
   *   values
   */
  isNotNaN(memoryResource?: MemoryResource) {
    return Series.new(this._col.isNotNaN(memoryResource));
  }

  /**
   * Round each floating-point value in this Series to the nearest integer.
   *
   * @param memoryResource Memory resource used to allocate the result Series's device memory.
   * @returns A Series of the same number of elements containing the result of the operation.
   */
  rint(memoryResource?: MemoryResource): Series<T> {
    return Series.new(this._col.rint(memoryResource));
  }

  _process_reduction(skipna = true, memoryResource?: MemoryResource): Series<T> {
    if (skipna == true) {
      return this.__construct(this._col.nans_to_nulls(memoryResource).drop_nulls());
    }
    return this.__construct(this._col);
  }
  /**
   * convert NaN values in the series with Null values,
   * while also updating the nullMask and nullCount values
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns updated Series with Null values
   */
  nansToNulls(memoryResource?: MemoryResource): Series<T> {
    return this.__construct(this._col.nans_to_nulls(memoryResource));
  }

  /**
   * drop NaN values from the column if column is of floating-type
   * values and contains NA values
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns column without NaN values
   */
  dropNaNs(memoryResource?: MemoryResource): Series<T> {
    return this.__construct(this._col.drop_nans(memoryResource));
  }

  /**
   * Return whether all elements are true in Series.
   *
   * @param skipna bool
   * Exclude NA/null values. If the entire row/column is NA and skipna is true, then the result will
   * be true, as for an empty row/column. If skipna is false, then NA are treated as true, because
   * these are not equal to zero.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   *
   * @returns true if all elements are true in Series, else false.
   */
  all(skipna = true, memoryResource?: MemoryResource) {
    if (skipna) {
      const ser_result = this.nansToNulls(memoryResource);
      if (ser_result.length == ser_result.nullCount) { return true; }
    }
    return this._col.all(memoryResource);
  }

  /**
   * Return whether any elements are true in Series.
   *
   * @param skipna bool
   * Exclude NA/null values. If the entire row/column is NA and skipna is true, then the result will
   * be true, as for an empty row/column. If skipna is false, then NA are treated as true, because
   * these are not equal to zero.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   *
   * @returns true if any elements are true in Series, else false.
   */
  any(skipna = true, memoryResource?: MemoryResource) {
    if (this.length == 0) { return false; }
    if (skipna) {
      const ser_result = this.nansToNulls(memoryResource);
      if (ser_result.length == ser_result.nullCount) { return false; }
    }
    return this._col.any(memoryResource);
  }

  /**
   * Compute the mean of all values in this Series.
   *
   * @param skipna The optional skipna if true drops NA and null values before computing reduction,
   * else if skipna is false, reduction is computed directly.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns The mean of all the values in this Series.
   */
  mean(skipna = true, memoryResource?: MemoryResource) {
    if (!skipna && this.nansToNulls().nullCount > 0) { return NaN; }
    return this._process_reduction(skipna, memoryResource)._col.mean(memoryResource);
  }

  /**
   * Compute the median of all values in this Series.
   *
   * @param skipna The optional skipna if true drops NA and null values before computing reduction,
   * else if skipna is false, reduction is computed directly.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns The median of all the values in this Series.
   */
  median(skipna = true, memoryResource?: MemoryResource) {
    if (!skipna && this.nansToNulls().nullCount > 0) { return NaN; }
    return this._process_reduction(skipna, memoryResource)._col.median(memoryResource);
  }

  /**
   * Compute the nunique of all values in this Series.
   *
   * @param dropna
   * If true, NA/null values will not contribute to the count of unique values. If false, they will
   * be included in the count.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns The number of unqiue values in this Series.
   */
  nunique(dropna = true, memoryResource?: MemoryResource) {
    return (dropna) ? this._col.nans_to_nulls(memoryResource).nunique(dropna, memoryResource)
                    : this._col.nunique(dropna, memoryResource);
  }
}

/**
 * A Series of 32-bit floating-point values in GPU memory.
 */
export class Float32Series extends FloatSeries<Float32> {
  /**
   * A Float32 view of the values in GPU memory.
   */
  get data() {
    return new Float32Buffer(this._col.data).subarray(this.offset, this.offset + this.length);
  }
}

/**
 * A Series of 64-bit floating-point values in GPU memory.
 */
export class Float64Series extends FloatSeries<Float64> {
  /**
   * A Float64 view of the values in GPU memory.
   */
  get data() {
    return new Float64Buffer(this._col.data).subarray(this.offset, this.offset + this.length);
  }
}
