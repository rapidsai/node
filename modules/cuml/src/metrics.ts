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

import {DataFrame, Numeric, Series} from '@rapidsai/cudf';

import {CUML} from './addon';
import {dataframe_to_series} from './utilities/array_utils';

/**
 * Expresses to what extent the local structure is retained in embedding. The score is defined in
 * the range [0, 1].
 * @param X original high dimensional dataset
 * @param embedded low dimesional embedding
 * @param n_neighbors Number of neighbors considered
 * @param batch_size It sets the number of samples that will be included in each batch
 * @returns Trustworthiness of the low-dimensional embedding
 */
export function trustworthinessSeries<T extends Numeric, R extends Numeric>(
  X: Series<T>, embedded: Series<R>, n_neighbors = 5, batch_size = 512): number {
  const n_samples  = X.length;
  const n_features = 1;

  const n_components = (embedded instanceof Series) ? 1 : embedded.numColumns;

  return CUML.trustworthiness(X.data.buffer,
                              embedded.data.buffer,
                              n_samples,
                              n_features,
                              n_components,
                              n_neighbors,
                              batch_size);
}

export function trustworthinessDF<T extends Numeric, R extends Numeric, K extends string>(
  X: DataFrame<{[P in K]: T}>,
  embedded: DataFrame<{[P in number]: R}>,
  n_neighbors = 5,
  batch_size  = 512): number {
  const n_samples  = X.numRows;
  const n_features = X.numColumns;

  const n_components = (embedded instanceof Series) ? 1 : embedded.numColumns;

  return CUML.trustworthiness(dataframe_to_series(X).data.buffer,
                              dataframe_to_series(embedded).data.buffer,
                              n_samples,
                              n_features,
                              n_components,
                              n_neighbors,
                              batch_size);
}
