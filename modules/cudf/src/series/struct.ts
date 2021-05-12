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

import {MemoryResource} from '@rapidsai/rmm';
import * as arrow from 'apache-arrow';

import {Column} from '../column';
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
   *
   * @example
   * ```typescript
   * import {Series} = require('@rapidsai/cudf');
   * import * as arrow from 'apache-arrow';
   *
   * const vec = arrow.Vector.from({
   *   values: [{ x: 0, y: 3 }, { x: 1, y: 4 }, { x: 2, y: 5 }],
   *   type: new arrow.Struct([
   *     arrow.Field.new({ name: 'x', type: new arrow.Int32 }),
   *     arrow.Field.new({ name: 'y', type: new arrow.Int32 })
   *   ]),
   * });
   * const a = Series.new(vec);
   *
   * a.getChild('x') // Int32Series [0, 1, 2]
   * a.getChild('y') // Int32Series [3, 4, 5]
   * ```
   */
  // TODO: Account for this.offset
  getChild<P extends keyof T>(name: P): Series<T[P]> {
    return Series.new(
      this._col.getChild<T[P]>(this.type.children.findIndex((f) => f.name === name)));
  }

  /** @ignore */
  protected __construct(col: Column<Struct<T>>) {
    return new StructSeries(Object.assign(col, {type: fixNames(this.type, col.type)}));
  }
}

Object.defineProperty(StructSeries.prototype, '__construct', {
  writable: false,
  enumerable: false,
  configurable: true,
  value: (StructSeries.prototype as any).__construct,
});

function fixNames<T extends DataType>(lhs: T, rhs: T) {
  if (lhs.children && rhs.children && lhs.children.length && rhs.children.length) {
    lhs.children.forEach(({name, type}, idx) => {
      rhs.children[idx] = arrow.Field.new({name, type: fixNames(type, rhs.children[idx].type)});
    });
  }
  return rhs;
}
