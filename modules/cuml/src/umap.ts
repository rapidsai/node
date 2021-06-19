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
import {
  Bool8,
  DataFrame,
  Float32,
  Int16,
  Int32,
  Int64,
  Int8,
  Integral,
  Series,
  Uint16,
  Uint32,
  Uint64,
  Uint8
} from '@rapidsai/cudf';
import {DeviceBuffer} from '@rapidsai/rmm';
import {compareTypes} from 'apache-arrow/visitor/typecomparator';

import {CUMLLogLevels, MetricType} from './mappings';
import {UMAPBase, UMAPInterface, UMAPParams} from './umap_base';
import {dataframeToSeries, seriesToDataframe} from './utilities/array_utils';

export type Numeric = Integral|Float32;

const allowedTypes = [
  new Int8,
  new Int16,
  new Int32,
  new Uint8,
  new Uint16,
  new Uint32,
  new Float32,
  new Bool8,
  new Int64,
  new Uint64
];

export type outputType = 'dataframe'|'series'|'devicebuffer';

class Embeddings<T extends Numeric> {
  private _embeddings: MemoryData|DeviceBuffer;
  private _dType: T;
  public nSamples: number;
  public nFeatures: number;

  constructor(_embeddings: MemoryData|DeviceBuffer, nSamples: number, nFeatures: number, dType: T) {
    this._embeddings = _embeddings;
    this.nSamples    = nSamples;
    this.nFeatures   = nFeatures;
    this._dType      = dType;
  }
  public asSeries() { return Series.new({type: this._dType, data: this._embeddings}); }
  public asDataFrame() { return seriesToDataframe(this.asSeries(), this.nSamples, this.nFeatures); }
  public asDeviceBuffer() { return this._embeddings; }
}

export class UMAP<T extends Series<Numeric> = any> {
  private _umap: UMAPInterface;
  private _embeddings: MemoryData|DeviceBuffer;
  private _features: T;
  private _nFeatures: number;
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
  constructor(input: UMAPParams,
              embeddings  = new DeviceBuffer(),
              features: T = Series.new({type: new Float32, data: []}) as T,
              nFeatures   = 1) {
    this._umap       = new UMAPBase(input);
    this._embeddings = embeddings;
    this._features   = features;
    this._nFeatures  = nFeatures;
  }

  protected _generate_embeddings<D extends Numeric>(nSamples: number, dtype: D) {
    return Series.sequence({type: dtype, size: nSamples * this._umap.nComponents, init: 0, step: 0})
      .data.buffer;
  }

  // throw runtime error if type if float64
  protected _check_type<D extends Numeric>(features: D) {
    if (!allowedTypes.some((type) => compareTypes(features, type))) {
      throw new Error('Expected input to be of type in [Integral, Float32] but got Float64');
    }
  }

  private _resolveType(convertDType: boolean, value: number|bigint|null|undefined) {
    return (convertDType) ? new Float32 : (typeof value == 'bigint') ? new Int64 : new Float32;
  }
  /**
   * Fit features into an embedded space
   *
   * @param features cuDF Series containing floats or doubles in the format [x1, y1, z1, x2, y2,
   *   z2...] for features x, y & z.
   *
   * @param target cuDF Series containing target values
   * @param convertDType When set to True, the method will automatically convert the inputs to
   *   float32
   *
   * @param nFeatures number of properties in the input features, if features is of the format
   *   [x1,y1,x2,y2...]
   * @returns UMAP object with update embeddings
   */
  fitSeries<R extends Series<Numeric>, B extends boolean>(features: T,
                                                          target: null|R,
                                                          convertDType: B,
                                                          nFeatures = 1) {
    // runtime type check
    this._check_type(features.type);
    const nSamples   = Math.floor(features.length / nFeatures);
    this._embeddings = this._generate_embeddings(nSamples, features.type);
    let options      = {
      features: features.data,
      featuresType: features.type,
      nSamples: nSamples,
      nFeatures: nFeatures,
      convertDType: convertDType,
      embeddings: this._embeddings
    };
    if (target !== null) {
      options = {...options, ...{ target: target.data, targetType: target.type }};
    }
    return new UMAP(this.getUMAPParams(), this._umap.fit(options), features, nFeatures);
  }

  /**
   * Fit features into an embedded space
   * @param features Dense or sparse matrix containing floats or doubles. Acceptable dense formats:
   *   cuDF DataFrame
   *
   * @param target cuDF Series containing target values
   * @param convertDType When set to True, the method will automatically convert the inputs to
   *   float32
   * @returns UMAP object with update embeddings
   */
  fitDataFrame<K extends string, R extends Series<Numeric>,>(features: DataFrame<{[P in K]: T['type']}>,
                                                    target: null|R,
                                                    convertDType: boolean) {
    // runtime type check
    this._check_type(features.get(features.names[0]).type);
    return this.fitSeries(
      dataframeToSeries(features) as T, target, convertDType, features.numColumns);
  }

  /**
   * Fit features into an embedded space
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
   * target = [color1, color2] // len(target) = len(features)
   * ```
   * @param convertDType When set to True, the method will automatically convert the inputs to
   *   float32
   * @param nFeatures number of properties in the input features, if features is of the format
   *   [x1,y1,x2,y2...]
   *
   * @returns UMAP object with update embeddings
   */
  fit(features: (number|bigint|null|undefined)[],
      target: (number|bigint|null|undefined)[]|null,
      convertDType: boolean,
      nFeatures = 1) {
    return this.fitSeries(
      Series.new({type: this._resolveType(convertDType, features[0]), data: features}) as T,
      (target == null)
        ? null
        : Series.new({type: this._resolveType(convertDType, target[0]), data: target}),
      convertDType,
      nFeatures);
  }

  /**
   * Fit features into an embedded space and return that transformed output.
   *
   *  There is a subtle difference between calling fit_transform(features) and calling
   * fit().transform(). Calling fit_transform(features) will train the embeddings on features and
   * return the embeddings. Calling fit(features).transform(features) will train the embeddings on
   * features and then run a second optimization.
   *
   *
   * @param features Dense or sparse matrix containing floats or doubles. Acceptable dense formats:
   *   cuDF Series
   * @param target cuDF Series containing target values
   * @param convertDType When set to True, the method will automatically convert the inputs to
   *   float32
   * @param nFeatures number of properties in the input features, if features is of the format
   *   [x1,y1,x2,y2...]
   * @param returnType Desired output type of results and attributes of the estimators
   *
   *  @returns Embedding of the data in low-dimensional space as per the outputType parameter while
   *   initializing UMAP, which can be converted to any of the following types: DataFrame, Series,
   * DeviceBuffer.
   *  ```typescript
   *  embeddings.asDataFrame(); // returns DataFrame<{number: Series<Numeric>}>
   *  embeddings.asSeries(); // returns Series<Numeric>
   *  embeddings.asDeviceBuffer(); //returns rmm.DeviceBuffer
   *  ```
   */
  fitTransformSeries<R extends Series<Numeric>, B extends boolean>(features: T,
                                                                   target: null|R,
                                                                   convertDType: B,
                                                                   nFeatures = 1) {
    const nSamples = Math.floor(features.length / nFeatures);
    const dtype    = (convertDType) ? new Float32 : features.type as T['type'];

    return new Embeddings(this.fitSeries(features, target, convertDType, nFeatures)._embeddings,
                          nSamples,
                          this.nComponents,
                          dtype);
  }

  /**
   * Fit features into an embedded space and return that transformed output.
   *
   *  There is a subtle difference between calling fit_transform(features) and calling
   * fit().transform(). Calling fit_transform(features) will train the embeddings on features and
   * return the embeddings. Calling fit(features).transform(features) will train the embeddings on
   * features and then run a second optimization.
   *
   *
   * @param features Dense or sparse matrix containing floats or doubles. Acceptable dense formats:
   *   cuDF DataFrame
   * @param target cuDF Series containing target values
   * @param convertDType When set to True, the method will automatically convert the inputs to
   *   float32
   * @param returnType Desired output type of results and attributes of the estimators
   *
   *  @returns Embedding of the data in low-dimensional space as per the outputType parameter while
   *   initializing UMAP, which can be converted to any of the following types: DataFrame, Series,
   * DeviceBuffer.
   *  ```typescript
   *  embeddings.asDataFrame(); // returns DataFrame<{number: Series<Numeric>}>
   *  embeddings.asSeries(); // returns Series<Numeric>
   *  embeddings.asDeviceBuffer(); //returns rmm.DeviceBuffer
   *  ```
   */
  fitTransformDataFrame<K extends string, R extends Series<Numeric>, B extends boolean>(
    features: DataFrame<{[P in K]: T['type']}>, target: null|R, convertDType: B) {
    return this.fitTransformSeries(
      dataframeToSeries(features) as T, target, convertDType, features.numColumns);
  }

  /**
   * Fit features into an embedded space and return that transformed output.
   *
   *  There is a subtle difference between calling fit_transform(features) and calling
   * fit().transform(). Calling fit_transform(features) will train the embeddings on features and
   * return the embeddings. Calling fit(features).transform(features) will train the embeddings on
   * features and then run a second optimization.
   *
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
   * ```typescript
   * // For a sample dataset of colors, with properties r,g and b:
   * target = [color1, color2] // len(target) = len(features)
   * ```

   * @param convertDType When set to True, the method will automatically convert the inputs to
   *   float32
   * @param nFeatures number of properties in the input features, if features is of the format
   *   [x1,y1,x2,y2...]
   * @param returnType Desired output type of results and attributes of the estimators
   *
   *  @returns Embedding of the data in low-dimensional space as per the outputType parameter while
   *   initializing UMAP, which can be converted to any of the following types: DataFrame, Series,
   * DeviceBuffer.
   *  ```typescript
   *  embeddings.asDataFrame(); // returns DataFrame<{number: Series<Numeric>}>
   *  embeddings.asSeries(); // returns Series<Numeric>
   *  embeddings.asDeviceBuffer(); //returns rmm.DeviceBuffer
   *  ```
   */
  fitTransform<D extends number|bigint, R extends number|bigint, B extends boolean>(
    features: (D|null|undefined)[],
    target: (R|null|undefined)[]|null,
    convertDType: B,
    nFeatures = 1) {
    return this.fitTransformSeries(
      Series.new({type: this._resolveType(convertDType, features[0]), data: features}) as T,
      (target == null)
        ? null
        : Series.new({type: this._resolveType(convertDType, target[0]), data: target}),
      convertDType,
      nFeatures);
  }

  /**
   * Transform features into the existing embedded space and return that transformed output.
   *
   * @param convertDType When set to True, the method will automatically convert the inputs to
   *   float32
   * @param nFeatures number of properties in the input features, if features is of the format
   *   [x1,y1,x2,y2...]
   *  @returns Embedding of the data in low-dimensional space as per the outputType parameter while
   *   initializing UMAP, which can be converted to any of the following types: DataFrame, Series,
   * DeviceBuffer.
   *  ```typescript
   *  embeddings.asDataFrame(); // returns DataFrame<{number: Series<Numeric>}>
   *  embeddings.asSeries(); // returns Series<Numeric>
   *  embeddings.asDeviceBuffer(); //returns rmm.DeviceBuffer
   *  ```
   */
  transform(convertDType: boolean, transformedEmbeddings: DeviceBuffer|null = null) {
    // runtime type check
    this._check_type(this._features.type);

    const nSamples   = Math.floor(this._features.length / this._nFeatures);
    const embeddings = (transformedEmbeddings == null)
                         ? this._generate_embeddings(nSamples, this._features.type)
                         : transformedEmbeddings;

    const result = this._umap.transform({
      features: this._features.data,
      featuresType: this._features.type,
      nSamples: nSamples,
      nFeatures: this._nFeatures,
      convertDType: convertDType,
      embeddings: this._embeddings,
      transformed: embeddings
    });

    const dtype = (convertDType) ? new Float32 : this._features.type;
    return new Embeddings(result, nSamples, this.nComponents, dtype);
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
