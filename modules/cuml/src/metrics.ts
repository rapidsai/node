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
// export function trustworthiness<T extends Numeric, R extends Numeric>(
//   X: Series<T>, embedded: Series<R>, n_neighbors?: number, batch_size?: number): number;

// export function trustworthiness<T extends Numeric, R extends Numeric>(
//   X: Series<T>, embedded: Series<R>, n_neighbors: number): number;

// export function trustworthiness<T extends Numeric, R extends Numeric>(X: Series<T>,
//                                                                       embedded: Series<R>):
//                                                                       number;

// export function trustworthiness<T extends Numeric, R extends Numeric, K extends string>(
//   X: DataFrame<{[P in K]: T}>,
//   embedded: DataFrame<{[P in number]: R}>,
//   n_neighbors: number,
//   batch_size: number): number;

// export function trustworthiness<T extends Numeric, R extends Numeric, K extends string>(
//   X: DataFrame<{[P in K]: T}>, embedded: DataFrame<{[P in number]: R}>, n_neighbors: number):
//   number;

// export function trustworthiness<T extends Numeric, R extends Numeric, K extends string>(
//   X: DataFrame<{[P in K]: T}>, embedded: DataFrame<{[P in number]: R}>): number;

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
