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

import {
  Column,
  FloatingPoint,
  Series,
} from '@rapidsai/cudf';

import {
  lonLatToCartesian,
} from './addon';
import {
  SeriesPair,
} from './node_cuspatial';

export function convertLonLatToCartesian<T extends FloatingPoint>(
  centerX: number, centerY: number, lonPoints: Column<T>, latPoints: Column<T>): SeriesPair<T> {
  const result = lonLatToCartesian(centerX, centerY, lonPoints, latPoints);
  const x      = Series.new(result.x);
  const y      = Series.new(result.y);
  return {x: x, y: y};
}
