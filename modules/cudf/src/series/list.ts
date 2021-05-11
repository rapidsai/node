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
import {DataType, Int32, List} from '../types/dtypes';

/**
 * A Series of lists of values.
 */
export class ListSeries<T extends DataType> extends Series<List<T>> {
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
   * Series of integer offsets for each list
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * import * as arrow from 'apache-arrow';
   *
   * const vec = arrow.Vector.from({
   *   values: [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
   *   type: new arrow.List(arrow.Field.new({ name: 'ints', type: new arrow.Int32 })),
   * });
   * const a = Series.new(vec);
   *
   * a.offsets // Int32Series [0, 3, 6, 9]
   * ```
   */
  // TODO: account for this.offset
  get offsets() { return Series.new(this._col.getChild<Int32>(0)); }

  /**
   * Series containing the elements of each list
   * @example
   * ```typescript
   * import {Series} from '@rapidsai/cudf';
   * import * as arrow from 'apache-arrow';
   *
   * const vec = arrow.Vector.from({
   *   values: [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
   *   type: new arrow.List(arrow.Field.new({ name: 'ints', type: new arrow.Int32 })),
   * });
   * const a = Series.new(vec);
   *
   * a.elements // Int32Series [0, 1, 2, 3, 4, 5, 6, 7, 8]
   * ```
   */
  // TODO: account for this.offset
  get elements(): Series<T> { return Series.new(this._col.getChild<T>(1)); }

  /**
   * Return a value at the specified index to host memory
   *
   * @param index the index in this Series to return a value for
   *
   * @example
   * ```typescript
   * import {Series} from "@rapidsai/cudf";
   *
   * // Series<List<Float64>>
   * Series.new([[1, 2], [3]]).getValue(0) // Series([1, 2])
   *
   * // Series<List<Utf8String>>
   * Series.new([["foo", "bar"], ["test"]]).getValue(1) // Series(["test"])
   *
   * // Series<List<Bool8>>
   * Series.new([[false, true], [true]]).getValue(2) // throws index out of bounds error
   * ```
   */
  getValue(index: number) {
    const value = this._col.getValue(index);
    return value === null ? null : Series.new(value);
  }

  /** @ignore */
  protected __construct(col: Column<List<T>>) {
    return new ListSeries(Object.assign(col, {type: fixNames(this.type, col.type)}));
  }
}

Object.defineProperty(ListSeries.prototype, '__construct', {
  writable: false,
  enumerable: false,
  configurable: true,
  value: (ListSeries.prototype as any).__construct,
});

function fixNames<T extends DataType>(lhs: T, rhs: T) {
  if (lhs.children && rhs.children && lhs.children.length && rhs.children.length) {
    lhs.children.forEach(({name, type}, idx) => {
      rhs.children[idx] = arrow.Field.new({name, type: fixNames(type, rhs.children[idx].type)});
    });
  }
  return rhs;
}
