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

import {MemoryData} from '@rapidsai/cuda';
import {
  Bool8,
  DataFrame,
  Float32,
  Float64,
  Int16,
  Int32,
  Int64,
  Int8,
  Numeric,
  Series,
  Uint16,
  Uint32,
  Uint64,
  Uint8
} from '@rapidsai/cudf';
import {DeviceBuffer} from '@rapidsai/rmm';
import {compareTypes} from 'apache-arrow/visitor/typecomparator';

import {COO} from './coo';
import {CUMLLogLevels, MetricType} from './mappings';
import {UMAP as UMAPBase, UMAPParams} from './umap_base';
import {seriesToDataFrame} from './utilities/array_utils';

const allowedTypes = [
  new Int8,
  new Int16,
  new Int32,
  new Uint8,
  new Uint16,
  new Uint32,
  new Float32,
  new Float64,
  new Bool8,
  new Int64,
  new Uint64
];

export type outputType = 'dataframe'|'series'|'devicebuffer';

class Embeddings {
  protected _embeddings: DeviceBuffer;
  protected nFeatures: number;

  constructor(_embeddings: DeviceBuffer, nFeatures: number) {
    this._embeddings = _embeddings;
    this.nFeatures   = nFeatures;
  }

  public asSeries() { return Series.new({type: new Float32, data: this._embeddings}); }
  public asDataFrame() { return seriesToDataFrame(this.asSeries(), this.nFeatures); }
  public asDeviceBuffer() { return this._embeddings; }
}

export class UMAP {
  protected _umap: UMAPBase;
  protected _embeddings: DeviceBuffer|undefined;

  /**
   * Initialize a UMAP object
   * @param input: UMAPParams
   * @param outputType: 'dataframe'|'series'|'devicebuffer'
   *
   * @example:
   * ```typescript
   * import {UMAP} from 'rapidsai/cuml';
   *
   * const umap = new UMAP({nComponents:2}, 'dataframe');
   * ```
   */
  constructor(input: UMAPParams = {}) { this._umap = new UMAPBase(input); }

  protected _generate_embeddings(nSamples: number) {
    return Series
      .sequence({
        type: new Float32,
        size: nSamples * this._umap.nComponents,
        init: 0,
        step: 0,
      })
      ._col.data;
  }

  // throw runtime error if type isn't Integral | Float32
  protected _check_type<D extends Numeric>(features: D) {
    if (!allowedTypes.some((type) => compareTypes(features, type))) {
      throw new Error(
        `Expected input to be of type in [Integral, Float32] but got ${features.toString()}`);
    }
  }

  /**
   * Fit features into an embedded space
   *
   * @note This method will automatically convert the inputs to float32
   *
   * @param features cuDF Series containing floats or doubles in the format [x1, y1, z1, x2, y2,
   *   z2...] for features x, y & z.
   *
   * @param target cuDF Series containing target values
   *
   * ```typescript
   * // For a sample dataset of colors, with properties r,g and b:
   * target = [color1, color2] // len(target) = nFeatures
   * ```
   *
   * @param nFeatures number of properties in the input features, if features is of the format
   *   [x1,y1,x2,y2...]
   *
   * @returns FittedUMAP object with updated embeddings
   */
  fitSeries<T extends Series<Numeric>, R extends Series<Numeric>>(features: T,
                                                                  target?: R|null,
                                                                  nFeatures = 1) {
    // runtime type check
    this._check_type(features.type);
    target && this._check_type(target.type);

    const f = features.cast(new Float32)._col;
    const t = target?.cast(new Float32)._col;

    const nSamples = Math.floor(f.length / nFeatures);

    const options = {
      nSamples,
      nFeatures,
      features: f,
      target: t,
      embeddings: this._generate_embeddings(nSamples),
    };

    const graph      = this._umap.graph(options);
    const embeddings = this._umap.fit({...options, graph});

    return new FittedUMAP(this.getUMAPParams(), graph, embeddings);
  }

  /**
   * Fit features into an embedded space
   *
   * @note This method will automatically convert the inputs to float32
   *
   * @param features Dense or sparse matrix containing floats or doubles. Acceptable dense formats:
   *   cuDF DataFrame
   *
   * @param target cuDF Series containing target values
   *
   * ```typescript
   * // For a sample dataset of colors, with properties r,g and b:
   * target = [color1, color2] // len(target) = nFeatures
   * ```
   *
   * @returns FittedUMAP object with updated embeddings
   */
  fitDataFrame<T extends Numeric, R extends Numeric, K extends string>(features:
                                                                         DataFrame<{[P in K]: T}>,
                                                                       target?: Series<R>) {
    // runtime type check
    features.names.forEach((name) => this._check_type(features.types[name]));
    return this.fitSeries(features.interleaveColumns(), target, features.numColumns);
  }

  /**
   * Fit features into an embedded space.
   *
   * @note This method will automatically convert the inputs to float32
   *
   * @param features array containing floats or doubles in the format [x1, y1, z1, x2, y2, z2...]
   *   for features x, y & z.
   *
   * ```typescript
   * // For a sample dataset of colors, with properties r,g and b:
   * features = [
   *   ...Object.values({ r: xx1, g: xx2, b: xx3 }),
   *   ...Object.values({ r: xx4, g: xx5, b: xx6 }),
   * ] // [xx1, xx2, xx3, xx4, xx5, xx6]
   * ```
   *
   * @param target array containing target values
   *
   * ```typescript
   * // For a sample dataset of colors, with properties r,g and b:
   * target = [color1, color2] // len(target) = nFeatures
   * ```
   *
   * @param nFeatures number of properties in the input features, if features is of the format
   *   [x1, y1, x2, y2...]
   *
   * @returns FittedUMAP object with updated embeddings
   */
  fitArray(features: MemoryData, target?: MemoryData|null, nFeatures?: number): FittedUMAP;
  fitArray(features: DeviceBuffer, target?: DeviceBuffer|null, nFeatures?: number): FittedUMAP;
  fitArray<T extends Series<Numeric>, R extends Series<Numeric>>(features: T,
                                                                 target?: R|null,
                                                                 nFeatures?: number): FittedUMAP;
  fitArray(features: (bigint|number|null|undefined)[],
           target?: (bigint|number|null|undefined)[]|null,
           nFeatures?: number): FittedUMAP;
  fitArray(features: any, target?: any|null, nFeatures = 1) {
    return this.fitSeries(Series.new(features), target && Series.new(target), nFeatures);
  }

  setUMAPParams(input: UMAPParams) { this._umap = new UMAPBase(input); }

  getUMAPParams() {
    return {
      nNeighbors: this.nNeighbors,
      nComponents: this.nComponents,
      nEpochs: this.nEpochs,
      learningRate: this.learningRate,
      minDist: this.minDist,
      spread: this.spread,
      setOpMixRatio: this.setOpMixRatio,
      localConnectivity: this.localConnectivity,
      repulsionStrength: this.repulsionStrength,
      negativeSampleRate: this.negativeSampleRate,
      transformQueueSize: this.transformQueueSize,
      verbosity: this.verbosity,
      a: this.a,
      b: this.b,
      initialAlpha: this.initialAlpha,
      init: this.init,
      targetNNeighbors: this.targetNNeighbors,
      targetMetric: this.targetMetric,
      targetWeight: this.targetWeight,
      randomState: this.randomState,
    } as UMAPParams;
  }

  /**
   * @returns Embeddings in low-dimensional space in dtype format, which can be converted to any
   * of the following types: DataFrame, Series, DeviceBuffer.
   *  ```typescript
   *  // returns DataFrame<{[K extends number]: Series<Float32>}>
   *  getEmbeddings(new Float64).asDataFrame();
   *  // returns Series<Float32>
   *  getEmbeddings(new Int32).asSeries();
   *  // returns rmm.DeviceBuffer
   *  getEmbeddings(new UInt32).asDeviceBuffer();
   *  ```
   */
  getEmbeddings() {
    return new Embeddings(this._embeddings || new DeviceBuffer(), this.nComponents);
  }

  /**
   * @returns Embeddings in low-dimensional space in float32 format, which can be converted to any
   * of the following types: DataFrame, Series, DeviceBuffer.
   *  ```typescript
   *  // returns DataFrame<{[K extends number]: Series<Float32>}>
   *  embeddings.asDataFrame();
   *  // returns Series<Float32>
   *  embeddings.asSeries();
   *  // returns rmm.DeviceBuffer
   *  embeddings.asDeviceBuffer();
   *  ```
   */
  get embeddings() {
    return new Embeddings(this._embeddings || new DeviceBuffer(), this.nComponents);
  }

  get nNeighbors(): number { return this._umap.nNeighbors; }
  get nComponents(): number { return this._umap.nComponents; }
  get nEpochs(): number { return this._umap.nEpochs; }
  get learningRate(): number { return this._umap.learningRate; }
  get minDist(): number { return this._umap.minDist; }
  get spread(): number { return this._umap.spread; }
  get setOpMixRatio(): number { return this._umap.setOpMixRatio; }
  get localConnectivity(): number { return this._umap.localConnectivity; }
  get repulsionStrength(): number { return this._umap.repulsionStrength; }
  get negativeSampleRate(): number { return this._umap.negativeSampleRate; }
  get transformQueueSize(): number { return this._umap.transformQueueSize; }
  get a(): number { return this._umap.a; }
  get b(): number { return this._umap.b; }
  get initialAlpha(): number { return this._umap.initialAlpha; }
  get init(): number { return this._umap.init; }
  get targetNNeighbors(): number { return this._umap.targetNNeighbors; }
  get targetWeight(): number { return this._umap.targetWeight; }
  get randomState(): number { return this._umap.randomState; }
  get targetMetric(): string { return MetricType[this._umap.targetMetric]; }
  get verbosity(): string { return CUMLLogLevels[this._umap.verbosity]; }
}

export class FittedUMAP extends UMAP {
  protected _graph: COO;
  protected _embeddings: DeviceBuffer;

  /**
   * @private
   */
  constructor(input: UMAPParams, graph: COO, embeddings: DeviceBuffer) {
    super(input);
    this._graph      = graph;
    this._embeddings = embeddings;
  }

  /**
   * Transform features into the existing embedded space and return that transformed output.
   *
   * @note This method will automatically convert the inputs to float32
   *
   * @param nFeatures number of properties in the input features, if features is of the format
   *   [x1,y1,x2,y2...]
   *
   * @returns Transformed `features` into the existing embedded space and return an `Embeddings`
   * instancewhich can be converted to any of the following types: DataFrame, Series, DeviceBuffer.
   *  ```typescript
   *  transformSeries(...).asDataFrame(); // returns DataFrame<{number: Series<Numeric>}>
   *  transformSeries(...).asSeries(); // returns Series<Numeric>
   *  transformSeries(...).asDeviceBuffer(); //returns rmm.DeviceBuffer
   *  ```
   */
  transformSeries<T extends Series<Numeric>>(features: T, nFeatures = 1) {
    // runtime type check
    this._check_type(features.type);

    const f = features.cast(new Float32)._col;

    const nSamples = Math.floor(f.length / nFeatures);

    const result = this._umap.transform({
      features: f,
      nSamples: nSamples,
      nFeatures: nFeatures,
      embeddings: this._embeddings,
      transformed: this._generate_embeddings(nSamples)
    });

    return new Embeddings(result, this.nComponents);
  }

  /**
   * Transform features into the existing embedded space and return that transformed output.
   *
   * @note This method will automatically convert the inputs to float32
   *
   * @param nFeatures number of properties in the input features, if features is of the format
   *   [x1,y1,x2,y2...]
   *
   * @returns Transformed `features` into the existing embedded space and return an `Embeddings`
   * instance which can be converted to any of the following types: DataFrame, Series, DeviceBuffer
   *  ```typescript
   *  transformDataFrame(...).asDataFrame(); // returns DataFrame<{number: Series<Numeric>}>
   *  transformDataFrame(...).asSeries(); // returns Series<Numeric>
   *  transformDataFrame(...).asDeviceBuffer(); //returns rmm.DeviceBuffer
   *  ```
   */
  transformDataFrame<T extends Numeric, K extends string>(features: DataFrame<{[P in K]: T}>) {
    // runtime type check
    features.names.forEach((name) => this._check_type(features.types[name]));
    return this.transformSeries(features.interleaveColumns(), features.numColumns);
  }

  /**
   * Transform features into the existing embedded space and return that transformed output.
   *
   * @note This method will automatically convert the inputs to float32
   *
   * @param nFeatures number of properties in the input features, if features is of the format
   *   [x1,y1,x2,y2...]
   *
   * @returns Transformed `features` into the existing embedded space and return an `Embeddings`
   * instance which can be converted to any of the following types: DataFrame, Series, DeviceBuffer.
   *  ```typescript
   *  transformArray(...).asDataFrame(); // returns DataFrame<{number: Series<Numeric>}>
   *  transformArray(...).asSeries(); // returns Series<Numeric>
   *  transformArray(...).asDeviceBuffer(); //returns rmm.DeviceBuffer
   *  ```
   */
  transformArray(features: MemoryData, nFeatures?: number): Embeddings;
  transformArray(features: DeviceBuffer, nFeatures?: number): Embeddings;
  transformArray<T extends Series<Numeric>>(features: T, nFeatures?: number): Embeddings;
  transformArray(features: (bigint|number|null|undefined)[], nFeatures?: number): Embeddings;
  transformArray(features: any, nFeatures = 1) {
    return this.transformSeries(Series.new(features), nFeatures);
  }

  /**
   * Refine features into existing embedded space as base
   *
   * @note This method will automatically convert the inputs to float32
   *
   * @param features cuDF Series containing floats or doubles in the format [x1, y1, z1, x2, y2,
   *   z2...] for features x, y & z.
   *
   * @param nFeatures number of properties in the input features, if features is of the format
   *   [x1,y1,x2,y2...]
   */
  refineSeries<T extends Series<Numeric>>(features: T, nFeatures = 1) {
    // runtime type check
    this._check_type(features.type);

    const f        = features.cast(new Float32)._col;
    const nSamples = Math.floor(f.length / nFeatures);

    this._umap.refine(
      {features: f, nSamples, nFeatures, graph: this._graph, embeddings: this._embeddings});
  }

  /**
   * Refine features into existing embedded space as base
   *
   * @note This method will automatically convert the inputs to float32
   *
   * @param features Dense or sparse matrix containing floats or doubles. Acceptable dense formats:
   *   cuDF DataFrame
   */
  refineDataFrame<T extends Numeric, K extends string>(features: DataFrame<{[P in K]: T}>) {
    // runtime type check
    features.names.forEach((name) => this._check_type(features.types[name]));
    this.refineSeries(features.interleaveColumns(), features.numColumns);
  }

  /**
   * Refine features into existing embedded space as base
   *
   * @note This method will automatically convert the inputs to float32
   *
   * @param features array containing floats or doubles in the format [x1, y1, z1, x2, y2, z2...]
   *   for features x, y & z.
   *
   * ```typescript
   * // For a sample dataset of colors, with properties r,g and b:
   * features = [
   *   ...Object.values({ r: xx1, g: xx2, b: xx3 }),
   *   ...Object.values({ r: xx4, g: xx5, b: xx6 }),
   * ] // [xx1, xx2, xx3, xx4, xx5, xx6]
   * ```
   *
   * @param nFeatures number of properties in the input features, if features is of the format
   *   [x1, y1, x2, y2...]
   *
   */
  refineArray(features: MemoryData, nFeatures?: number): void;
  refineArray(features: DeviceBuffer, nFeatures?: number): void;
  refineArray<T extends Series<Numeric>>(features: T, nFeatures?: number): void;
  refineArray(features: (bigint|number|null|undefined)[], nFeatures?: number): void;
  refineArray(features: any, nFeatures = 1) { this.refineSeries(Series.new(features), nFeatures); }
}
