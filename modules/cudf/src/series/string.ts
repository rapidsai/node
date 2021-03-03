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

import {MemoryResource} from '@nvidia/rmm';
import * as arrow from 'apache-arrow';

import {Series} from '../series';
import {DataType, Int32, Uint8, Utf8String} from '../types/dtypes'

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
   * Series of integer offsets for each string
   */
  // TODO: Account for this.offset
  get offsets() { return Series.new(this._col.getChild<Int32>(0)); }
  /**
   * Series containing the utf8 characters of each string
   */
  // TODO: Account for this.offset
  get data() { return Series.new(this._col.getChild<Uint8>(1)); }
}
