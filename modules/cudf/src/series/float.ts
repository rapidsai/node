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
import {MemoryResource} from '@nvidia/rmm';

import {Series} from '../series';
import {Float32, Float64, FloatingPoint, SeriesType} from '../types';

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
  rint(memoryResource?: MemoryResource): SeriesType<T> {
    return Series.new(this._col.rint(memoryResource));
  }
}

/**
 * A Series of 32-bit floating-point values in GPU memory.
 */
export class Float32Series extends FloatSeries<Float32> {
  /**
   * A Float32 view of the values in GPU memory.
   */
  get data() { return new Float32Buffer(this._col.data).subarray(0, this.length); }
}

/**
 * A Series of 64-bit floating-point values in GPU memory.
 */
export class Float64Series extends FloatSeries<Float64> {
  /**
   * A Float64 view of the values in GPU memory.
   */
  get data() { return new Float64Buffer(this._col.data).subarray(0, this.length); }
}
