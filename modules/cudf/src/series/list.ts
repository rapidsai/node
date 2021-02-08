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

import {Series} from '../series';
import {DataType, Int32, List, SeriesType} from '../types';

/**
 * A Series of lists of values.
 */
export class ListSeries<T extends DataType> extends Series<List<T>> {
  /**
   * Series of integer offsets for each list
   */
  get offsets() { return Series.new(this._col.getChild<Int32>(0)); }

  /**
   * Series containing the elements of each list
   */
  get elements(): SeriesType<T> { return Series.new(this._col.getChild<T>(1)); }
}
