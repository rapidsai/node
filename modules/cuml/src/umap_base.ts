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
  targetMetric?: MetricType,
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
