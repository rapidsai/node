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

import {Column, FloatingPoint, Integral} from '@rapidsai/cudf';
import {DeviceBuffer, MemoryResource} from '@rapidsai/rmm';

/** @ignore */
export declare const _cpp_exports: any;

export declare class GraphCOO {
  constructor(src: Column<Integral|FloatingPoint>,
              dst: Column<Integral|FloatingPoint>,
              options?: {directedEdges?: boolean});

  numEdges(): number;
  numNodes(): number;

  forceAtlas2(options: {
    memoryResource?: MemoryResource,
    positions?: DeviceBuffer,
    numIterations?: number,
    outboundAttraction?: boolean,
    linLogMode?: boolean,
    // preventOverlap?: boolean, ///< not implemented in cuGraph yet
    edgeWeightInfluence?: number,
    jitterTolerance?: number,
    barnesHutTheta?: number,
    scalingRatio?: number,
    strongGravityMode?: boolean,
    gravity?: number,
    verbose?: boolean,
  }): DeviceBuffer;
}
