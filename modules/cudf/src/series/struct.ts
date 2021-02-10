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
import {DataType, Struct} from '../types/dtypes';
import {TypeMap} from '../types/mappings';

/**
 * A Series of structs.
 */
export class StructSeries<T extends TypeMap> extends Series<Struct<T>> {
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
   * Return a child series by name.
   *
   * @param name Name of the Series to return.
   */
  getChild<P extends keyof T>(name: P): Series<T[P]> {
    return Series.new(
      this._col.getChild<T[P]>(this.type.children.findIndex((f) => f.name === name)));
  }
}
