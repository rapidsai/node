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

import {Uint8ClampedBuffer} from '@rapidsai/cuda';
import {MemoryResource} from '@rapidsai/rmm';

import {Series, StringSeries} from '../series';
import {Bool8, Int64, Uint8} from '../types/dtypes';

import {NumericSeries} from './numeric';

/**
 * A Series of 1-byte boolean values in GPU memory.
 */
export class Bool8Series extends NumericSeries<Bool8> {
  _castAsString(memoryResource?: MemoryResource): StringSeries {
    return StringSeries.new(this._col.stringsFromBooleans(memoryResource));
  }

  /**
   * A Uint8 view of the values in GPU memory.
   */
  get data() {
    return new Uint8ClampedBuffer(this._col.data).subarray(this.offset, this.offset + this.length);
  }

  toBitMask() { return this._col.boolsToMask()[0]; }

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
   * const a = Series.new([true, false, true])
   *
   * a.cumulativeMax() // {true, true, true}
   * ```
   */
  cumulativeMax(skipNulls = true, memoryResource?: MemoryResource) {
    return this.__construct(this._prepare_scan_series(skipNulls).cumulativeMax(memoryResource));
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
   * const a = Series.new([true, false, true])
   *
   * a.cumulativeMin() // {true, false, false}
   * ```
   */
  cumulativeMin(skipNulls = true, memoryResource?: MemoryResource) {
    return this.__construct(this._prepare_scan_series(skipNulls).cumulativeMin(memoryResource));
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
   * const a = Series.new([true, false, true])
   *
   * a.cumulativeProduct() // {1, 0, 0}
   * ```
   */
  cumulativeProduct(skipNulls = true, memoryResource?: MemoryResource) {
    return Series.new(
      this._prepare_scan_series(skipNulls).cast(new Uint8).cumulativeProduct(memoryResource));
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
   * const a = Series.new([true, false, true])
   *
   * a.cumulativeSum() // {1n, 1n, 2n}
   * ```
   */
  cumulativeSum(skipNulls = true, memoryResource?: MemoryResource) {
    return Series.new(
      this._prepare_scan_series(skipNulls).cast(new Int64).cumulativeSum(memoryResource));
  }

  /** @inheritdoc */
  min(skipNulls = true, memoryResource?: MemoryResource) {
    return super.min(skipNulls, memoryResource) as boolean;
  }

  /** @inheritdoc */
  max(skipNulls = true, memoryResource?: MemoryResource) {
    return super.max(skipNulls, memoryResource) as boolean;
  }

  /** @inheritdoc */
  minmax(skipNulls = true, memoryResource?: MemoryResource) {
    return super.minmax(skipNulls, memoryResource) as [boolean, boolean];
  }

  /** @inheritdoc */
  sum(skipNulls = true, memoryResource?: MemoryResource) {
    return super.sum(skipNulls, memoryResource) as number;
  }

  /** @inheritdoc */
  product(skipNulls = true, memoryResource?: MemoryResource) {
    return super.product(skipNulls, memoryResource) as number;
  }

  /** @inheritdoc */
  sumOfSquares(skipNulls = true, memoryResource?: MemoryResource) {
    return super.sumOfSquares(skipNulls, memoryResource) as number;
  }
}
