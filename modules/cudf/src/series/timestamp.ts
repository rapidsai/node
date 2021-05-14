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

import {Int32Buffer, Int64Buffer} from '@nvidia/cuda';
import {MemoryResource} from '@rapidsai/rmm';
import {Series} from '../series';
import {
  DataType,
  Timestamp,
  TimestampDay,
  TimestampMicrosecond,
  TimestampMillisecond,
  TimestampNanosecond,
  TimestampSecond
} from '../types/dtypes';

export abstract class TimestampSeries<T extends Timestamp> extends Series<T> {
  /**
   * Casts the values to a new dtype (similar to `static_cast` in C++).
   *
   * @param dataType The new dtype.
   * @param memoryResource The optional MemoryResource used to allocate the result Series's device
   *   memory.
   * @returns Series of same size as the current Series containing result of the `cast` operation.
   * @example
   * ```typescript
   * import {Series, Bool8, Int32} from '@rapidsai/cudf';
   *
   * const a = Series.new({type:new Int32, data: [1,0,1,0]});
   *
   * a.cast(new Bool8); // Bool8Series [true, false, true, false];
   * ```
   */
  cast<R extends DataType>(dataType: R, memoryResource?: MemoryResource): Series<R> {
    return Series.new(this._col.cast(dataType, memoryResource));
  }
}

export class TimestampDaySeries extends TimestampSeries<TimestampDay> {
  get data() {
    return new Int32Buffer(this._col.data).subarray(this.offset, this.offset + this.length);
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
   * Series.new([1, 2, 3]).getValue(2) // 3
   * Series.new([1, 2, 3]).getValue(3) // throws index out of bounds error
   * ```
   */
  getValue(index: number) {
    const val = this._col.getValue(index);
    return val === null ? null : new Date(val * 86400000);
  }
}

export class TimestampSecondSeries extends TimestampSeries<TimestampSecond> {
  get data() {
    return new Int64Buffer(this._col.data).subarray(this.offset, this.offset + this.length);
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
   * Series.new([1, 2, 3]).getValue(2) // 3
   * Series.new([1, 2, 3]).getValue(3) // throws index out of bounds error
   * ```
   */
  getValue(index: number) {
    const val = this._col.getValue(index);
    return val === null ? null : new Date(Number(val * 1000n));
  }
}

export class TimestampMicrosecondSeries extends TimestampSeries<TimestampMicrosecond> {
  get data() {
    return new Int64Buffer(this._col.data).subarray(this.offset, this.offset + this.length);
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
   * Series.new([1, 2, 3]).getValue(2) // 3
   * Series.new([1, 2, 3]).getValue(3) // throws index out of bounds error
   * ```
   */
  getValue(index: number) {
    const val = this._col.getValue(index);
    return val === null ? null : new Date(Number(val / 1000n));
  }
}

export class TimestampMillisecondSeries extends TimestampSeries<TimestampMillisecond> {
  get data() {
    return new Int64Buffer(this._col.data).subarray(this.offset, this.offset + this.length);
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
   * Series.new([1, 2, 3]).getValue(2) // 3
   * Series.new([1, 2, 3]).getValue(3) // throws index out of bounds error
   * ```
   */
  getValue(index: number) {
    const val = this._col.getValue(index);
    return val === null ? null : new Date(Number(val));
  }
}

export class TimestampNanosecondSeries extends TimestampSeries<TimestampNanosecond> {
  get data() {
    return new Int64Buffer(this._col.data).subarray(this.offset, this.offset + this.length);
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
   * Series.new([1, 2, 3]).getValue(2) // 3
   * Series.new([1, 2, 3]).getValue(3) // throws index out of bounds error
   * ```
   */
  getValue(index: number) {
    const val = this._col.getValue(index);
    return val === null ? null : new Date(Number(val / 1000000n));
  }
}
