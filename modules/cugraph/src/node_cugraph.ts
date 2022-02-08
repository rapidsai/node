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

import {MemoryData, MemoryView} from '@rapidsai/cuda';
import {Column, FloatingPoint, Integral} from '@rapidsai/cudf';
import {DeviceBuffer, MemoryResource} from '@rapidsai/rmm';

/** @ignore */
export declare const _cpp_exports: any;

export interface ForceAtlas2Options {
  /**
   * Optional buffer of initial vertex positions.
   */
  positions?: MemoryData|MemoryView|DeviceBuffer;
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

export declare interface CUGraph {
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
  forceAtlas2(options: ForceAtlas2Options): DeviceBuffer;

  /**
   * @summary Compute the total number of edges incident to a vertex (both in and out edges).
   */
  degree(): Column;
}

export declare class GraphCOO implements CUGraph {
  constructor(src: Column<Integral|FloatingPoint>,
              dst: Column<Integral|FloatingPoint>,
              options?: {directedEdges?: boolean});

  /** @inheritdoc */
  numEdges(): number;

  /** @inheritdoc */
  numNodes(): number;

  /** @inheritdoc */
  forceAtlas2(options: ForceAtlas2Options): DeviceBuffer;

  /** @inheritdoc */
  degree(): Column;
}
