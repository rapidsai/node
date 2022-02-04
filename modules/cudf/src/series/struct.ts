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

import {Series} from '../series';
import {Struct} from '../types/dtypes';
import {TypeMap} from '../types/mappings';

/**
 * A Series of structs.
 */
export class StructSeries<T extends TypeMap> extends Series<Struct<T>> {
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
    return value === null
             ? null
             : this.type.children.reduce(
                 (xs, f, i) => ({...xs, [f.name]: value.getColumnByIndex(i).getValue(0)}), {});
  }
}
