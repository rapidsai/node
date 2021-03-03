// Copyright (c) 2020, NVIDIA CORPORATION.
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

import {Column, Int32} from '@nvidia/cudf';
import {DeviceBuffer, MemoryResource} from '@nvidia/rmm';

export interface GraphCOOConstructor {
  readonly prototype: GraphCOO;
  new(src: Column<Int32>, dst: Column<Int32>, options?: {directedEdges?: boolean}): GraphCOO;
}

// eslint-disable-next-line @typescript-eslint/no-redeclare
interface GraphCOO {
  readonly numEdges: number;
  readonly numNodes: number;

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

export {GraphCOO} from './addon';
