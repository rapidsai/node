import {DeviceBuffer} from '@rapidsai/rmm/build/js';
import {CUML} from './addon';

export type UMAPParams = {
  n_neighbors?: number,
  n_components?: number,
  n_epochs?: number,
  learning_rate?: number,
  min_dist?: number,
  spread?: number,
  set_op_mix_ratio?: number,
  local_connectivity?: number,
  repulsion_strength?: number,
  negative_sample_rate?: number,
  transform_queue_size?: number,
  // verbosity?: number,
  a?: number,
  b?: number,
  initial_alpha?: number,
  init?: number,
  target_n_neighbors?: number,
  // target_metric?: number,
  target_weights?: number,
  multicore_implem?: boolean,
  optim_batch_size?: number
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
  // readonly verbosity: number;
  readonly a: number;
  readonly b: number;
  readonly initialAlpha: number;
  readonly init: number;
  readonly targetNNeighbors: number;
  // readonly targetMetric: number;
  readonly targetWeights: number;
  readonly multicoreImplem: number;
  readonly optimBatchSize: number;

  fit(X: DeviceBuffer,
      n_samples: number,
      n_features: number,
      y: DeviceBuffer|null,
      knnIndices: DeviceBuffer|null,
      knnDists: DeviceBuffer|null,
      convertDType: boolean): void;

  transform(X: DeviceBuffer,
            n_samples: number,
            n_features: number,
            knnIndices: DeviceBuffer|null,
            knnDists: DeviceBuffer|null,
            convertDType: boolean): DeviceBuffer;

  getEmbeddings(): DeviceBuffer;
}

export const UMAPBase: UMAPConstructor = CUML.UMAP;
