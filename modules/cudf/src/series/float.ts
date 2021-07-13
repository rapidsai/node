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

import {Column} from '../column';
import {Scalar} from '../scalar';
import {Series} from '../series';
import {Bool8, Float32, Float64, FloatingPoint, Int32} from '../types/dtypes';

import {NumericSeries} from './numeric';

/**
 * A base class for Series of 32 or 64-bit floating-point values in GPU memory.
 */
abstract class FloatSeries<T extends FloatingPoint> extends NumericSeries<T> {
  /**
   * Creates a Series of `BOOL8` elements where `true` indicates the value is `NaN` and `false`
   * indicates the value is valid.
   *
   * @param memoryResource Memory resource used to allocate the result Column's device memory.
   * @returns A non-nullable Series of `BOOL8` elements with `true` representing `NaN` values.
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   *
   * // Float64Series
   * Series.new([1, NaN, 3]).isNaN() // [false, true, false]
   * ```
   */
  isNaN(memoryResource?: MemoryResource) { return Series.new(this._col.isNaN(memoryResource)); }

  /**
   * Creates a Series of `BOOL8` elements indicating the absence of `NaN` values in a
   * column of floating point values. The output element at row `i` is `false` if the element in
   * `input` at row i is `NaN`, else `true`
   *
   * @param memoryResource Memory resource used to allocate the result Series's device memory.
   * @returns A non-nullable Series of `BOOL8` elements with `true` representing `NAN`
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

  _process_reduction(skipNulls = true, memoryResource?: MemoryResource): Series<T> {
    if (skipNulls == true) {
      return this.__construct(this._col.nansToNulls(memoryResource).dropNulls());
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
    return this.__construct(this._col.nansToNulls(memoryResource));
  }

  /**
   * drop NaN values from the column if column is of floating-type
   * values and contains NA values
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   * @returns column without NaN values
   */
  dropNaNs(memoryResource?: MemoryResource): Series<T> {
    return this.__construct(this._col.dropNans(memoryResource));
  }

  /**
   * Replace NaN values with a scalar value.
   *
   * @param value The value to use in place of NaNs.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   */
  replaceNaNs(value: T['scalarType'], memoryResource?: MemoryResource): Series<T>;

  /**
   * Replace NaN values with the corresponding elements from another Series.
   *
   * @param value The Series to use in place of NaNs.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   */
  replaceNaNs(value: Series<T>, memoryResource?: MemoryResource): Series<T>;

  replaceNaNs(value: T['scalarType']|Series<T>, memoryResource?: MemoryResource): Series<T> {
    if (value instanceof Series) {
      return Series.new(this._col.replaceNaNs(value._col as Column<T>, memoryResource));
    } else {
      return Series.new(
        this._col.replaceNaNs(new Scalar<T>({type: this.type, value}), memoryResource));
    }
  }

  /**
   * Return whether all elements are true in Series.
   *
   * @param skipNulls bool
   * Exclude NA/null values. If the entire row/column is NA and skipNulls is true, then the result
   * will be true, as for an empty row/column. If skipNulls is false, then NA are treated as true,
   * because these are not equal to zero.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   *
   * @returns true if all elements are true in Series, else false.
   */
  all(skipNulls = true, memoryResource?: MemoryResource) {
    if (skipNulls) {
      const ser_result = this.nansToNulls(memoryResource);
      if (ser_result.length == ser_result.nullCount) { return true; }
    }
    return this._col.all(memoryResource);
  }

  /**
   * Return whether any elements are true in Series.
   *
   * @param skipNulls bool
   * Exclude NA/null values. If the entire row/column is NA and skipNulls is true, then the result
   * will be true, as for an empty row/column. If skipNulls is false, then NA are treated as true,
   * because these are not equal to zero.
   * @param memoryResource The optional MemoryResource used to allocate the result Column's device
   *   memory.
   *
   * @returns true if any elements are true in Series, else false.
   */
  any(skipNulls = true, memoryResource?: MemoryResource) {
    if (this.length == 0) { return false; }
    if (skipNulls) {
      const ser_result = this.nansToNulls(memoryResource);
      if (ser_result.length == ser_result.nullCount) { return false; }
    }
    return this._col.any(memoryResource);
  }

  protected _prepare_scan_series(skipNulls: boolean) {
    const data = this.nansToNulls();

    if (skipNulls) { return data; }

    if (!data.hasNulls) { return data; }

    const index = Series.sequence({type: new Int32, size: data.length, step: 1, init: 0});

    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    const first = index.filter(data.isNull()).getValue(0)!;
    const slice =
      Series.sequence({type: new Int32, size: data.length - first, step: 1, init: first});

    const copy = data.cast(data.type);
    const mask = [...index.cast(new Bool8).fill(true).scatter(false, slice)];
    copy.setNullMask(mask as any);

    return copy;
  }

  /**
   * Compute the cumulative max of all values in this Series.
   *
   * @param skipNulls The optional skipNulls if true drops NA and null values before computing
   *   reduction,
   * else if skipNulls is false, reduction is computed directly.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns The cumulative max of all the values in this Series.
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([4, 2, 5, 1, 1])
   *
   * a.cumulativeMax() // {4, 4, 5, 5, 5}
   * ```
   */
  cumulativeMax(skipNulls = true, memoryResource?: MemoryResource) {
    const result_series = this._prepare_scan_series(skipNulls);
    return Series.new(result_series._col.cumulativeMax(memoryResource) as Column<T>);
  }

  /**
   * Compute the cumulative min of all values in this Series.
   *
   * @param skipNulls The optional skipNulls if true drops NA and null values before computing
   *   reduction,
   * else if skipNulls is false, reduction is computed directly.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns The cumulative min of all the values in this Series.
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([4, 2, 5, 1, 1])
   *
   * a.cumulativeMin() // {4, 2, 2, 1, 1}
   * ```
   */
  cumulativeMin(skipNulls = true, memoryResource?: MemoryResource) {
    const result_series = this._prepare_scan_series(skipNulls);
    return Series.new(result_series._col.cumulativeMin(memoryResource) as Column<T>);
  }

  /**
   * Compute the cumulative product of all values in this Series.
   *
   * @param skipNulls The optional skipNulls if true drops NA and null values before computing
   *   reduction,
   * else if skipNulls is false, reduction is computed directly.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns The cumulative product of all the values in this Series.
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([4, 2, 5, 1, 1])
   *
   * a.cumulativeProduct() // {4, 8, 40, 40, 40}
   * ```
   */
  cumulativeProduct(skipNulls = true, memoryResource?: MemoryResource) {
    const result_series = this._prepare_scan_series(skipNulls);
    return Series.new(result_series._col.cumulativeProduct(memoryResource) as Column<T>);
  }

  /**
   * Compute the cumulative sum of all values in this Series.
   *
   * @param skipNulls The optional skipNulls if true drops NA and null values before computing
   *   reduction,
   * else if skipNulls is false, reduction is computed directly.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns The cumulative sum of all the values in this Series.
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([4, 2, 5, 1, 1])
   *
   * a.cumulativeSum() // {4, 6, 11, 12, 13}
   * ```
   */
  cumulativeSum(skipNulls = true, memoryResource?: MemoryResource) {
    const result_series = this._prepare_scan_series(skipNulls);
    return Series.new(result_series._col.cumulativeSum(memoryResource) as Column<T>);
  }

  /**
   * Compute the mean of all values in this Series.
   *
   * @param skipNulls The optional skipNulls if true drops NA and null values before computing
   *   reduction,
   * else if skipNulls is false, reduction is computed directly.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns The mean of all the values in this Series.
   */
  mean(skipNulls = true, memoryResource?: MemoryResource) {
    if (!skipNulls && this.nansToNulls().nullCount > 0) { return NaN; }
    return this._process_reduction(skipNulls, memoryResource)._col.mean(memoryResource);
  }

  /**
   * Compute the median of all values in this Series.
   *
   * @param skipNulls The optional skipNulls if true drops NA and null values before computing
   *   reduction,
   * else if skipNulls is false, reduction is computed directly.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns The median of all the values in this Series.
   */
  median(skipNulls = true, memoryResource?: MemoryResource) {
    if (!skipNulls && this.nansToNulls().nullCount > 0) { return NaN; }
    return this._process_reduction(skipNulls, memoryResource)._col.median(memoryResource);
  }

  /**
   * Compute the nunique of all values in this Series.
   *
   * @param dropna
   * If true, NA/null values will not contribute to the count of unique values. If false, they
   * will be included in the count.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns The number of unqiue values in this Series.
   */
  nunique(dropna = true, memoryResource?: MemoryResource) {
    return (dropna) ? this._col.nansToNulls(memoryResource).nunique(dropna, memoryResource)
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
