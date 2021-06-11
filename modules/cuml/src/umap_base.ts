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

import {MemoryData} from '@nvidia/cuda';
import {DeviceBuffer} from '@rapidsai/rmm';

import {CUML} from './addon';
import {CUMLLogLevels, MetricType} from './mappings';

export type UMAPParams = {
  nNeighbors?: number,
  nComponents?: number,
  nEpochs?: number,
  learningRate?: number,
  minDist?: number,
  spread?: number,
  setOpMixRatio?: number,
  localConnectivity?: number,
  repulsionStrength?: number,
  negativeSampleRate?: number,
  transformQueueSize?: number,
  verbosity?: keyof typeof CUMLLogLevels,
  a?: number,
  b?: number,
  initialAlpha?: number,
  init?: number,
  targetNNeighbors?: number,
  targetMetric?: keyof typeof MetricType,
  targetWeight?: number,
  randomState?: number,
};

interface UMAPConstructor {
  new(options?: UMAPParams): UMAPInterface;
}

export interface UMAPInterface {
  readonly nNeighbors: number;
  readonly nComponents: number;
  readonly nEpochs: number;
  readonly learningRate: number;
  readonly minDist: number;
  readonly spread: number;
  readonly setOpMixRatio: number;
  readonly localConnectivity: number;
  readonly repulsionStrength: number;
  readonly negativeSampleRate: number;
  readonly transformQueueSize: number;
  readonly verbosity: number;
  readonly a: number;
  readonly b: number;
  readonly initialAlpha: number;
  readonly init: number;
  readonly targetNNeighbors: number;
  readonly targetMetric: number;
  readonly targetWeight: number;
  readonly randomState: number;

  fit(X: MemoryData|DeviceBuffer,
      n_samples: number,
      n_features: number,
      y: MemoryData|DeviceBuffer|null,
      knnIndices: DeviceBuffer|null,
      knnDists: DeviceBuffer|null,
      convertDType: boolean,
      embeddings: MemoryData|DeviceBuffer): DeviceBuffer;

  transform(X: MemoryData|DeviceBuffer,
            n_samples: number,
            n_features: number,
            knnIndices: MemoryData|DeviceBuffer|null,
            knnDists: MemoryData|DeviceBuffer|null,
            convertDType: boolean,
            embeddings: MemoryData|DeviceBuffer,
            transformed: MemoryData|DeviceBuffer): DeviceBuffer;
}
export const UMAPBase: UMAPConstructor = CUML.UMAP;
