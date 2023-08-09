// Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

import {Memory, MemoryData, MemoryView} from '@rapidsai/cuda';
import {Column, Float32, Int32} from '@rapidsai/cudf';
import {DeviceBuffer, MemoryResource} from '@rapidsai/rmm';

/** @ignore */
export declare const _cpp_exports: any;

export declare class Graph {
  constructor(props: {
    src: Column<Int32>,
    dst: Column<Int32>,
    weight: Column<Float32>,
    directed?: boolean,
  });

  /**
   * @summary The number of edges in this Graph
   */
  numEdges(): number;

  /**
   * @summary The number of nodes in this Graph
   */
  numNodes(): number;

  /**
   * @summary ForceAtlas2 is a continuous graph layout algorithm for handy network visualization.
   *
   * @note Peak memory allocation occurs at 30*V.
   *
   * @param {ForceAtlas2Options} options
   *
   * @returns {DeviceBuffer} The new positions.
   */
  forceAtlas2<T extends ForceAtlas2Options<Memory>>(options: T): T['positions'];
  forceAtlas2<T extends ForceAtlas2Options<MemoryView>>(options: T): T['positions'];
  forceAtlas2<T extends ForceAtlas2Options<MemoryData|DeviceBuffer|void>>(options: T): DeviceBuffer;

  /**
   * @summary Compute the total number of edges incident to a vertex (both in and out edges).
   */
  degree(): Column<Int32>;

  /**
   * @summary Compute a clustering/partitioning of the given graph using the spectral modularity
   * maximization method.
   *
   * @param {SpectralClusteringOptions} options
   */
  spectralModularityMaximizationClustering(options: SpectralClusteringOptions): Column<Int32>;

  /**
   * @summary Compute a clustering/partitioning of the given graph using the spectral balanced cut
   * method.
   *
   * @param {SpectralClusteringOptions} options
   */
  spectralBalancedCutClustering(options: SpectralClusteringOptions): Column<Int32>;

  /**
   * @summary Compute the modularity score for a given partitioning/clustering. The assumption is
   * that "clustering" is the results from a call from a special clustering algorithm and contains
   * columns named "vertex" and "cluster".
   *
   * @param {number} num_clusters The number of clusters.
   * @param {Column<Int32>} clusters The Column of cluster ids.
   *
   * @returns {number} The computed modularity score
   */
  analyzeModularityClustering(num_clusters: number, clusters: Column<Int32>): number;

  /**
   * @summary Compute the edge cut score for a partitioning/clustering The assumption is that
   * "clustering" is the results from a call from a special clustering algorithm and contains
   * columns named "vertex" and "cluster".
   *
   * @param {number} num_clusters The number of clusters.
   * @param {Column<Int32>} clusters The Column of cluster ids.
   *
   * @returns {number} The computed edge cut score
   */
  analyzeEdgeCutClustering(num_clusters: number, clusters: Column<Int32>): number;

  /**
   * @summary Compute the ratio cut score for a partitioning/clustering.
   *
   * @param {number} num_clusters The number of clusters.
   * @param {Column<Int32>} clusters The Column of cluster ids.
   *
   * @returns {number} The computed ratio cut score
   */
  analyzeRatioCutClustering(num_clusters: number, clusters: Column<Int32>): number;
}

export interface ForceAtlas2Options<TPositions = void> {
  /**
   * Optional buffer of initial vertex positions.
   */
  positions: TPositions;
  /**
   * The maximum number of levels/iterations of the Force Atlas algorithm. When specified the
   * algorithm will terminate after no more than the specified number of iterations. No error occurs
   * when the algorithm terminates in this manner. Good short-term quality can be achieved with
   * 50-100 iterations. Above 1000 iterations is discouraged.
   */
  numIterations?: number;
  /**
   * Distributes attraction along outbound edges. Hubs attract less and thus are pushed to the
   * borders.
   */
  outboundAttraction?: boolean;
  /**
   * Switch Force Atlas model from lin-lin to lin-log. Makes clusters more tight.
   */
  linLogMode?: boolean;
  /**
   * Prevent nodes from overlapping.
   */
  // preventOverlap?: boolean, ///< not implemented in cuGraph yet
  /**
   * How much influence you give to the edges weight. 0 is "no influence" and 1 is "normal".
   */
  edgeWeightInfluence?: number;
  /**
   * How much swinging you allow. Above 1 discouraged. Lower gives less speed and more precision.
   */
  jitterTolerance?: number;
  /**
   * Float between 0 and 1. Tradeoff for speed (1) vs accuracy (0).
   */
  barnesHutTheta?: number;
  /**
   * How much repulsion you want. More makes a more sparse graph. Switching from regular mode to
   * LinLog mode needs a readjustment of the scaling parameter.
   */
  scalingRatio?: number;
  /**
   * Sets a force that attracts the nodes that are distant from the center more. It is so strong
   * that it can sometimes dominate other forces.
   */
  strongGravityMode?: boolean;
  /**
   * Attracts nodes to the center. Prevents islands from drifting away.
   */
  gravity?: number;
  /**
   * Output convergence info at each interation.
   */
  verbose?: boolean;
  memoryResource?: MemoryResource;
}

export interface SpectralClusteringOptions {
  /**
   * @summary Specifies the number of clusters to find
   */
  num_clusters: number;
  /**
   * @summary Specifies the number of eigenvectors to use. Must be less than or equal to
   * `num_clusters`. Default is 2.
   */
  num_eigen_vecs?: number;
  /**
   * @summary Specifies the tolerance to use in the eigensolver. Default is 0.00001.
   */
  evs_tolerance?: number;
  /**
   * @summary Specifies the maximum number of iterations for the eigensolver. Default is 100.
   */
  evs_max_iter?: number;
  /**
   * @summary Specifies the tolerance to use in the k-means solver. Default is 0.00001.
   */
  kmean_tolerance?: number;
  /**
   * @summary Specifies the maximum number of iterations for the k-means solver. Default is 100.
   */
  kmean_max_iter?: number;
}
