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
import {DataFrame, Float32, Float64, Int64, Integral, Series} from '@rapidsai/cudf';
import {DeviceBuffer} from '@rapidsai/rmm';

import {CUMLLogLevels, MetricType} from './mappings';
import {UMAPBase, UMAPInterface, UMAPParams} from './umap_base';
import {dataframeToSeries, seriesToDataframe} from './utilities/array_utils';

export type Numeric = Integral|Float32;

export type outputType = 'dataframe'|'series'|'devicebuffer';

export type returnTypeMap<T extends outputType, R extends Numeric> = {
  'dataframe': DataFrame<{[P in number]: R}>,
  'series': Series<R>,
  'devicebuffer': DeviceBuffer
}[T];

export class UMAP<O extends outputType = any> {
  public _umap: UMAPInterface;
  public _embeddings: MemoryData|DeviceBuffer;
  public outputType: O;

  constructor(input: UMAPParams, outputType: O) {
    this._umap       = new UMAPBase(input);
    this._embeddings = new DeviceBuffer();
    this.outputType  = outputType;
  }

  protected _generate_embeddings(nSamples: number, dtype: Numeric): MemoryData {
    return Series.sequence({type: dtype, size: nSamples * this._umap.nComponents, init: 0, step: 0})
      .data.buffer;
  }

  // throw runtime error if type if float64
  protected _check_type(X: Numeric) {
    if (X.compareTo(new Float64)) {
      throw new Error('Expected input to be of type in [Integral, Float32] but got Float64');
    }
  }

  protected _process_embeddings<T extends Numeric>(embeddings: MemoryData,
                                                   nSamples: number,
                                                   dtype: T): returnTypeMap<O, T> {
    if (this.outputType == 'dataframe') {
      return seriesToDataframe(
               Series.new({type: dtype, data: embeddings}), nSamples, this.nComponents) as
             returnTypeMap<O, T>;
    } else if (this.outputType == 'series') {
      return Series.new({type: dtype, data: embeddings}) as returnTypeMap<O, T>;
    } else {
      return embeddings as returnTypeMap<O, T>;
    }
  }
  /**
   * Fit X into an embedded space
   * @param X cuDF Series containing floats or doubles in the format [x1, y1, z1, x2, y2, z2...] for
   *   features x, y & z.
   * @param y cuDF Series containing target values
   * @param convertDType When set to True, the method will automatically convert the inputs to
   *   float32
   * @param nFeatures number of features in the input X, if X is of the format [x1,y1,x2,y2...]
   */
  fitSeries<T extends Series<Numeric>, R extends Series<Numeric>, B extends boolean>(
    X: T, y: null|R, convertDType: B, nFeatures = 1) {
    // runtime type check
    this._check_type(X.type);
    const nSamples   = Math.floor(X.length / nFeatures);
    this._embeddings = this._generate_embeddings(nSamples, X.type);
    let options      = {
      X: X.data,
      XType: X.type,
      nSamples: nSamples,
      nFeatures: nFeatures,
      convertDType: convertDType,
      embeddings: this._embeddings
    };
    if (y !== null) {
      options = {...options, ...{ y: y.data, yType: y.type }};
    }
    this._embeddings = this._umap.fit(options);
  }

  /**
   * Fit X into an embedded space
   * @param X Dense or sparse matrix containing floats or doubles. Acceptable dense formats: cuDF
   *   DataFrame
   * @param y cuDF Series containing target values
   * @param convertDType When set to True, the method will automatically convert the inputs to
   *   float32
   */
  fitDF<T extends Numeric, R extends Series<Numeric>, K extends string>(X: DataFrame<{[P in K]: T}>,
                                                                        y: null|R,
                                                                        convertDType: boolean) {
    // runtime type check
    this._check_type(X.get(X.names[0]).type);
    const nSamples   = X.numRows;
    const nFeatures  = X.numColumns;
    this._embeddings = this._generate_embeddings(nSamples, X.get(X.names[0]).type);

    let options = {
      X: dataframeToSeries(X).data,
      XType: X.get(X.names[0]).type,
      nSamples: nSamples,
      nFeatures: nFeatures,
      convertDType: convertDType,
      embeddings: this._embeddings
    };
    if (y !== null) {
      options = {...options, ...{ y: y.data, yType: y.type }};
    }
    this._embeddings = this._umap.fit(options);
  }

  /**
   * Fit X into an embedded space
   * @param X array containing floats or doubles in the format [x1, y1, z1, x2, y2, z2...] for
   *   features x, y & z.
   * @param y array containing target values
   * @param convertDType When set to True, the method will automatically convert the inputs to
   *   float32
   * @param nFeatures number of features in the input X, if X is of the format [x1,y1,x2,y2...]
   */
  fit(X: (number|bigint|null|undefined)[],
      y: (number|bigint|null|undefined)[]|null,
      convertDType: boolean,
      nFeatures = 1) {
    const nSamples   = Math.floor(X.length / nFeatures);
    const XDType     = (typeof X[0] == 'bigint') ? new Int64 : new Float32;
    this._embeddings = this._generate_embeddings(nSamples, XDType);
    let options      = {
      X: X,
      XType: XDType,
      nSamples: nSamples,
      nFeatures: nFeatures,
      convertDType: convertDType,
      embeddings: this._embeddings
    };
    if (y !== null) {
      options = {
        ...options,
        ...{ y: y, yType: (typeof y[0] == 'bigint') ? new Int64 : new Float32 }
      };
    }
    this._embeddings = this._umap.fit(options);
  }

  /**
   * Fit X into an embedded space and return that transformed output.
   *
   *  There is a subtle difference between calling fit_transform(X) and calling fit().transform().
   *  Calling fit_transform(X) will train the embeddings on X and return the embeddings. Calling
   *  fit(X).transform(X) will train the embeddings on X and then run a second optimization.
   *
   *
   * @param X Dense or sparse matrix containing floats or doubles. Acceptable dense formats: cuDF
   *   Series
   * @param y cuDF Series containing target values
   * @param convertDType When set to True, the method will automatically convert the inputs to
   *   float32
   * @param nFeatures number of features in the input X, if X is of the format [x1,y1,x2,y2...]
   * @param returnType Desired output type of results and attributes of the estimators
   *
   * @returns Embedding of the data in low-dimensional space
   */
  fitTransformSeries<T extends Series<Numeric>, R extends Series<Numeric>, B extends boolean>(
    X: T, y: null|R, convertDType: B, nFeatures = 1) {
    this.fitSeries(X, y, convertDType, nFeatures);
    const nSamples = Math.floor(X.length / nFeatures);
    const dtype    = (convertDType) ? new Float32 : X.type as T['type'];

    return this._process_embeddings(this._embeddings, nSamples, dtype) as
           B extends true ? returnTypeMap<O, Float32>: returnTypeMap<O, T['type']>;
  }

  /**
   * Fit X into an embedded space and return that transformed output.
   *
   *  There is a subtle difference between calling fit_transform(X) and calling fit().transform().
   *  Calling fit_transform(X) will train the embeddings on X and return the embeddings. Calling
   *  fit(X).transform(X) will train the embeddings on X and then run a second optimization.
   *
   *
   * @param X Dense or sparse matrix containing floats or doubles. Acceptable dense formats: cuDF
   *   DataFrame
   * @param y cuDF Series containing target values
   * @param convertDType When set to True, the method will automatically convert the inputs to
   *   float32
   * @param returnType Desired output type of results and attributes of the estimators
   *
   * @returns Embedding of the data in low-dimensional space
   */
  fitTransformDF<T extends Numeric, R extends Series<Numeric>, K extends string, B extends boolean>(
    X: DataFrame<{[P in K]: T}>, y: null|R, convertDType: B) {
    const nSamples = X.numRows;
    this.fitDF(X, y, convertDType);
    const dtype = X.get(X.names[0]).type as T;
    return this._process_embeddings(this._embeddings, nSamples, dtype) as
           B extends true ? returnTypeMap<O, Float32>: returnTypeMap<O, T>;
  }

  /**
   * Fit X into an embedded space and return that transformed output.
   *
   *  There is a subtle difference between calling fit_transform(X) and calling fit().transform().
   *  Calling fit_transform(X) will train the embeddings on X and return the embeddings. Calling
   *  fit(X).transform(X) will train the embeddings on X and then run a second optimization.
   *
   *
   * @param X array containing floats or doubles in the format [x1, y1, z1, x2, y2, z2...] for
   *   features x, y & z.
   * @param y array containing target values
   * @param convertDType When set to True, the method will automatically convert the inputs to
   *   float32
   * @param nFeatures number of features in the input X, if X is of the format [x1,y1,x2,y2...]
   * @param returnType Desired output type of results and attributes of the estimators
   *
   * @returns Embedding of the data in low-dimensional space
   */
  fitTransform<T extends number|bigint, R extends number|bigint, B extends boolean>(
    X: (T|null|undefined)[], y: (R|null|undefined)[]|null, convertDType: B, nFeatures = 1) {
    this.fit(X, y, convertDType, nFeatures);

    const nSamples = Math.floor(X.length / nFeatures);
    const dtype    = (convertDType)              ? new Float32
                     : (typeof X[0] == 'bigint') ? new Int64
                                                 : new Float32;

    return this._process_embeddings(this._embeddings, nSamples, dtype) as
           B extends true ? returnTypeMap<O, Float32>: returnTypeMap < O,
                     T extends number ? Float32 : Int64 > ;
  }

  /**
   * Transform X into the existing embedded space and return that transformed output.
   *
   * @param X Dense or sparse matrix containing floats or doubles. Acceptable dense formats: cuDF
   *   Series
   * @param convertDType When set to True, the method will automatically convert the inputs to
   *   float32
   * @param nFeatures number of features in the input X, if X is of the format [x1,y1,x2,y2...]
   * @param returnType Desired output type of results and attributes of the estimators
   *
   * @returns Embedding of the data in low-dimensional space
   */
  transformSeries<T extends Series<Numeric>, B extends boolean>(X: T,
                                                                convertDType: B,
                                                                nFeatures = 1) {
    // runtime type check
    this._check_type(X.type);

    const nSamples   = Math.floor(X.length / nFeatures);
    const embeddings = this._generate_embeddings(nSamples, X.type);

    const result = this._umap.transform({
      X: X.data.buffer,
      XType: X.type,
      nSamples: nSamples,
      nFeatures: nFeatures,
      convertDType: convertDType,
      embeddings: this._embeddings,
      transformed: embeddings
    });

    const dtype = (convertDType) ? new Float32 : X.type;
    return this._process_embeddings(result, nSamples, dtype) as
           B extends true ? returnTypeMap<O, Float32>: returnTypeMap<O, T['type']>;
  }

  /**
   * Transform X into the existing embedded space and return that transformed output.
   *
   * @param X Dense or sparse matrix containing floats or doubles. Acceptable dense formats: cuDF
   *   Series
   * @param convertDType When set to True, the method will automatically convert the inputs to
   *   float32
   * @param returnType Desired output type of results and attributes of the estimators
   *
   * @returns Embedding of the data in low-dimensional space
   */
  transformDF<T extends Numeric, K extends string, B extends boolean>(X: DataFrame<{[P in K]: T}>,
                                                                      convertDType: B) {
    // runtime type check
    this._check_type(X.get(X.names[0]).type);
    const nSamples   = X.numRows;
    const nFeatures  = X.numColumns;
    const embeddings = this._generate_embeddings(nSamples, X.get(X.names[0]).type);

    const result = this._umap.transform({
      X: dataframeToSeries(X).data.buffer,
      XType: X.get(X.names[0]).type,
      nSamples: nSamples,
      nFeatures: nFeatures,
      convertDType: convertDType,
      embeddings: this._embeddings,
      transformed: embeddings
    });
    const dtype  = (convertDType) ? new Float32 : X.get(X.names[0]).type;
    return this._process_embeddings(result, nSamples, dtype) as
           B extends true ? returnTypeMap<O, Float32>: returnTypeMap<O, T>;
  }

  /**
   * Transform X into the existing embedded space and return that transformed output.
   *
   * @param X array containing floats or doubles in the format [x1, y1, z1, x2, y2, z2...] for
   *   features x, y & z.
   * @param convertDType When set to True, the method will automatically convert the inputs to
   *   float32
   * @param nFeatures number of features in the input X, if X is of the format [x1,y1,x2,y2...]
   * @param returnType Desired output type of results and attributes of the estimators
   *
   * @returns Embedding of the data in low-dimensional space
   */
  transform<T extends number|bigint, B extends boolean>(X: (T|null|undefined)[],
                                                        convertDType: B,
                                                        nFeatures = 1) {
    const nSamples   = Math.floor(X.length / nFeatures);
    const XDtype     = (convertDType)              ? new Float32
                       : (typeof X[0] == 'bigint') ? new Int64
                                                   : new Float32;
    const embeddings = this._generate_embeddings(nSamples, XDtype);

    const result = this._umap.transform({
      X: X,
      XType: XDtype,
      nSamples: nSamples,
      nFeatures: nFeatures,
      convertDType: convertDType,
      embeddings: this._embeddings,
      transformed: embeddings
    });

    return this._process_embeddings(result, nSamples, XDtype) as
           B extends true ? returnTypeMap<O, Float32>: returnTypeMap < O,
                     T extends number ? Float32 : Int64 > ;
  }

  get embeddings(): Series<Float32> {
    return Series.new({type: new Float32, data: this._embeddings});
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
