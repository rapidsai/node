// Copyright (c) 2022, NVIDIA CORPORATION.
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

import {
  Column,
  FloatingPoint,
  Series,
} from '@rapidsai/cudf';
import {makePoints} from '@rapidsai/cuspatial';
import {MemoryResource} from '@rapidsai/rmm';

import {
  lonLatToCartesian,
} from './addon';

export function convertLonLatToCartesian<T extends FloatingPoint>(centerX: number,
                                                                  centerY: number,
                                                                  lonPoints: Series<T>,
                                                                  latPoints: Series<T>,
                                                                  memoryResource?: MemoryResource) {
  const {x, y} = lonLatToCartesian(
    centerX, centerY, lonPoints._col as Column<T>, latPoints._col as Column<T>, memoryResource);
  return makePoints(Series.new(x), Series.new(y));
}
