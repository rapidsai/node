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

import {
  DataFrame,
  Int32,
  Numeric,
  Series,
} from '@rapidsai/cudf';

/**
 * convert a dataframe to a single series to replicate conversion to a matrix in an organized
 * for {x1,y1,x2,y2...} for df = {x:[x1,x2], y: [y1,y2]}
 */
export function dataframeToSeries<T extends Numeric, K extends string>(
  input: DataFrame<{[P in K]: T}>) {
  return input.interleaveColumns();
}

/**
 * convert a series to a dataframe as per the number of componenet in umapparams
 * @param input
 * @param n_samples
 * @param nComponents
 * @returns DataFrame
 */
export function seriesToDataframe<T extends Numeric>(
  input: Series<T>, nSamples: number, nComponents: number): DataFrame<{[P in number]: T}> {
  let result = new DataFrame<{[P in number]: T}>({});
  for (let i = 0; i < nComponents; i++) {
    result = result.assign({
      [i]:
        input.gather(Series.sequence({type: new Int32, init: i, size: nSamples, step: nComponents}))
    });
  }
  return result;
}
