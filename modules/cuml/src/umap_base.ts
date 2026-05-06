// Copyright (c) 2021-2026, NVIDIA CORPORATION.
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

import {Column, Float32, Int64} from '@rapidsai/cudf';
import {DeviceBuffer} from '@rapidsai/rmm';

import * as CUML from './addon';
import {COO} from './coo';
import {CUMLLogLevels, MetricType} from './mappings';

/**
 * UMAPParams parameters from https://docs.rapids.ai/api/cuml/stable/api.html#cuml.UMAP
 */
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

export type FitProps = {
  nSamples: number,
  nFeatures: number,
  features: Column<Float32>,
  graph: COO,
  embeddings: DeviceBuffer,
  target?: Column<Float32>,
  knnIndices?: Column<Int64>,
  knnDists?: Column<Float32>,
};

export type RefineProps = {
  nSamples: number,
  nFeatures: number,
  features: Column<Float32>,
  graph: COO,
  embeddings: DeviceBuffer,
};

export type GetGraphProps = {
  nSamples: number,
  nFeatures: number,
  features: Column<Float32>,
  target?: Column<Float32>,
  knnIndices?: Column<Int64>,
  knnDists?: Column<Float32>,
};

export type TransformProps = {
  features: Column<Float32>,
  nSamples: number,
  nFeatures: number,
  embeddings: DeviceBuffer,
  transformed: DeviceBuffer,
  knnIndices?: Column<Int64>,
  knnDists?: Column<Float32>,
};

export interface UMAPConstructor {
  readonly prototype: UMAP;

  new(options?: UMAPParams): UMAP;
}

export interface UMAP {
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

  fit(options?: FitProps): DeviceBuffer;

  transform(options?: TransformProps): DeviceBuffer;

  refine(options?: RefineProps): DeviceBuffer;

  graph(options?: GetGraphProps): COO;
}

export const UMAP: UMAPConstructor = CUML.UMAP;
