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

import {DataFrame, Float32, Numeric, Series} from '@rapidsai/cudf';
import {DeviceBuffer} from '@rapidsai/rmm';

import * as CUML from './addon';
import {dataframeToSeries} from './utilities/array_utils';

/**
 * Expresses to what extent the local structure is retained in embedding. The score is defined in
 * the range [0, 1].
 * @param features original high dimensional dataset
 * @param embedded low dimesional embedding
 * @param nFeatures Number of features in features
 * @param nComponents Number of features in embedded
 * @param nNeighbors Number of neighbors considered
 * @param batch_size It sets the number of samples that will be included in each batch
 * @returns Trustworthiness of the low-dimensional embedding
 */
export function trustworthinessSeries<T extends Numeric, R extends Numeric>(
  features: Series<T>,
  embedded: Series<R>,
  nFeatures: number,
  nComponents = 2,
  nNeighbors  = 5,
  batch_size  = 512): number {
  const nSamples = Math.floor(features.length / nFeatures);

  return CUML.trustworthiness(features.data.buffer as DeviceBuffer,
                              features.type,
                              embedded.data.buffer as DeviceBuffer,
                              embedded.type,
                              nSamples,
                              nFeatures,
                              nComponents,
                              nNeighbors,
                              batch_size);
}

/**
 * Expresses to what extent the local structure is retained in embedding. The score is defined in
 * the range [0, 1].
 * @param features original high dimensional dataset
 * @param embedded low dimesional embedding
 * @param nNeighbors Number of neighbors considered
 * @param batch_size It sets the number of samples that will be included in each batch
 * @returns Trustworthiness of the low-dimensional embedding
 */
export function trustworthinessDataFrame<T extends Numeric, R extends Numeric, K extends string>(
  features: DataFrame<{[P in K]: T}>,
  embedded: DataFrame<{[P in number]: R}>,
  nNeighbors = 5,
  batch_size = 512): number {
  const nSamples  = features.numRows;
  const nFeatures = features.numColumns;

  const nComponents = embedded.numColumns;

  return CUML.trustworthiness(dataframeToSeries(features).data.buffer as DeviceBuffer,
                              features.get(features.names[0]).type,
                              dataframeToSeries(embedded).data.buffer as DeviceBuffer,
                              embedded.get(embedded.names[0]).type,
                              nSamples,
                              nFeatures,
                              nComponents,
                              nNeighbors,
                              batch_size);
}

/**
 * Expresses to what extent the local structure is retained in embedding. The score is defined in
 * the range [0, 1].
 * @param features original high dimensional dataset
 * @param embedded low dimesional embedding
 * @param nFeatures Number of features in features
 * @param nComponents Number of features in embedded
 * @param nNeighbors Number of neighbors considered
 * @param batch_size It sets the number of samples that will be included in each batch
 * @returns Trustworthiness of the low-dimensional embedding
 */
export function trustworthiness<T extends number|bigint>(features: (T|null|undefined)[],
                                                         embedded: (T|null|undefined)[],
                                                         nFeatures: number,
                                                         nComponents = 2,
                                                         nNeighbors  = 5,
                                                         batch_size  = 512): number {
  const nSamples = Math.floor(features.length / nFeatures);
  return CUML.trustworthiness(features,
                              new Float32,
                              embedded,
                              new Float32,
                              nSamples,
                              nFeatures,
                              nComponents,
                              nNeighbors,
                              batch_size);
}
