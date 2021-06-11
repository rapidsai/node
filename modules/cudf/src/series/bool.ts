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

import {Uint8ClampedBuffer} from '@nvidia/cuda';
import {MemoryResource} from '@rapidsai/rmm';

import {Series} from '../series';
import {Bool8, Int32, Int64} from '../types/dtypes';

import {NumericSeries} from './numeric';

/**
 * A Series of 1-byte boolean values in GPU memory.
 */
export class Bool8Series extends NumericSeries<Bool8> {
  /**
   * A Uint8 view of the values in GPU memory.
   */
  get data() {
    return new Uint8ClampedBuffer(this._col.data).subarray(this.offset, this.offset + this.length);
  }

  protected _prepare_scan_series(skipna: boolean) {
    if (skipna || !this.hasNulls) { return this; }

    const index = Series.sequence({type: new Int32, size: this.length, step: 1, init: 0});

    const first = index.filter(this.isNull()).getValue(0)!;
    const slice =
      Series.sequence({type: new Int32, size: this.length - first, step: 1, init: first});

    const copy = this.cast(this.type);
    const mask = [...index.cast(new Bool8).fill(true).scatter(false, slice)];
    copy.setNullMask(mask as any);

    return copy;
  }

  /**
   * Compute the cumulative max of all values in this Series.
   *
   * @param skipna The optional skipna if true drops NA and null values before computing reduction,
   * else if skipna is false, reduction is computed directly.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns The cumulative max of all the values in this Series.
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([true, false, true])
   *
   * a.cummax() // {true, true, true}
   * ```
   */
  cummax(skipna = true, memoryResource?: MemoryResource) {
    const result_series = this._prepare_scan_series(skipna);
    return Series.new(result_series._col.cummax(memoryResource));
  }

  /**
   * Compute the cumulative min of all values in this Series.
   *
   * @param skipna The optional skipna if true drops NA and null values before computing reduction,
   * else if skipna is false, reduction is computed directly.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns The cumulative min of all the values in this Series.
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([true, false, true])
   *
   * a.cummin() // {true, false, false}
   * ```
   */
  cummin(skipna = true, memoryResource?: MemoryResource) {
    const result_series = this._prepare_scan_series(skipna);
    return Series.new(result_series._col.cummin(memoryResource));
  }

  /**
   * Compute the cumulative product of all values in this Series.
   *
   * @param skipna The optional skipna if true drops NA and null values before computing
   *   reduction,
   * else if skipna is false, reduction is computed directly.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns The cumulative product of all the values in this Series.
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([true, false, true])
   *
   * a.cumprod() // {1n, 1n, 2n}
   * ```
   */
  cumprod(skipna = true, memoryResource?: MemoryResource) {
    const result_series = this._prepare_scan_series(skipna).cast(new Int64, memoryResource);
    return Series.new(result_series._col.cumprod(memoryResource));
  }

  /**
   * Compute the cumulative sum of all values in this Series.
   *
   * @param skipna The optional skipna if true drops NA and null values before computing
   *   reduction,
   * else if skipna is false, reduction is computed directly.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns The cumulative sum of all the values in this Series.
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * const a = Series.new([true, false, true])
   *
   * a.cumsum() // {1n, 0n, 0n}
   * ```
   */
  cumsum(skipna = true, memoryResource?: MemoryResource) {
    const result_series = this._prepare_scan_series(skipna).cast(new Int64, memoryResource);
    return Series.new(result_series._col.cumsum(memoryResource));
  }
}
