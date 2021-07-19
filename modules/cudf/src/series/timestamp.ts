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
  Timestamp,
  TimestampDay,
  TimestampMicrosecond,
  TimestampMillisecond,
  TimestampNanosecond,
  TimestampSecond
} from '../types/dtypes';

export abstract class TimestampSeries<T extends Timestamp> extends Series<T> {
  _castAsTimeStampDay(memoryResource?: MemoryResource): Series<TimestampDay> {
    return Series.new(this._col.cast(new TimestampDay, memoryResource));
  }
  _castAsTimeStampSecond(memoryResource?: MemoryResource): Series<TimestampSecond> {
    return Series.new(this._col.cast(new TimestampSecond, memoryResource));
  }
  _castAsTimeStampMillisecond(memoryResource?: MemoryResource): Series<TimestampMillisecond> {
    return Series.new(this._col.cast(new TimestampMillisecond, memoryResource));
  }
  _castAsTimeStampMicrosecond(memoryResource?: MemoryResource): Series<TimestampMicrosecond> {
    return Series.new(this._col.cast(new TimestampMicrosecond, memoryResource));
  }
  _castAsTimeStampNanosecond(memoryResource?: MemoryResource): Series<TimestampNanosecond> {
    return Series.new(this._col.cast(new TimestampNanosecond, memoryResource));
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
   * import {Series, TimestampDay} from "@rapidsai/cudf";
   *
   * // TimestampDaySeries
   * const s = Series.new({
   *   type: new TimestampDay,
   *   data: [new Date('May 13, 2021 16:38:30:100 GMT+00:00')]
   * });
   *
   * s.getValue(0) // 2021-05-13T00:00:00.000Z
   * ```
   */
  getValue(index: number) {
    const val = this._col.getValue(index);
    return val === null ? null : new Date(val);
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
   * import {Series, TimestampSecond} from "@rapidsai/cudf";
   *
   * // TimestampSecondSeries
   * const s = Series.new({
   *   type: new TimestampSecond,
   *   data: [new Date('May 13, 2021 16:38:30:100 GMT+00:00')]
   * });
   *
   * s.getValue(0) // 2021-05-13T16:38:30.000Z
   * ```
   */
  getValue(index: number) {
    const val = this._col.getValue(index);
    return val === null ? null : new Date(Number(val));
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
   * import {Series, TimestampMicrosecond} from "@rapidsai/cudf";
   *
   * // TimestampMicrosecondSeries
   * const s = Series.new({
   *   type: new TimestampMicrosecond,
   *   data: [new Date('May 13, 2021 16:38:30:100 GMT+00:00')]
   * });
   *
   * s.getValue(0) // 2021-05-13T16:38:30.100Z
   * ```
   */
  getValue(index: number) {
    const val = this._col.getValue(index);
    return val === null ? null : new Date(Number(val));
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
   * import {Series, TimestampMillisecond} from "@rapidsai/cudf";
   *
   * // TimestampMillisecondSeries
   * const s = Series.new({
   *   type: new TimestampMillisecond,
   *   data: [new Date('May 13, 2021 16:38:30:100 GMT+00:00')]
   * });
   *
   * s.getValue(0) // 2021-05-13T16:38:30.100Z
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
   * import {Series, TimestampNanosecond} from "@rapidsai/cudf";
   *
   * // TimestampNanosecondSeries
   * const s = Series.new({
   *   type: new TimestampNanosecond,
   *   data: [new Date('May 13, 2021 16:38:30:100 GMT+00:00')]
   * });
   *
   * s.getValue(0) // 2021-05-13T16:38:30.100Z
   * ```
   */
  getValue(index: number) {
    const val = this._col.getValue(index);
    return val === null ? null : new Date(Number(val));
  }
}
