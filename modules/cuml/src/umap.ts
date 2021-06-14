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
import {DataFrame, Float32, Numeric, Series} from '@rapidsai/cudf';
import {DeviceBuffer} from '@rapidsai/rmm';

import {CUMLLogLevels, MetricType} from './mappings';
import {UMAPBase, UMAPInterface, UMAPParams} from './umap_base';
import {dataframeToSeries, seriesToDataframe} from './utilities/array_utils';

type returnType = 'dataframe'|'series'|'devicebuffer';

export type returnTypeMap<T extends returnType, R extends Numeric> = {
  'dataframe': DataFrame<{[P in number]: R}>,
  'series': Series<R>,
  'devicebuffer': MemoryData|DeviceBuffer
}[T];

export class UMAP {
  public _umap: UMAPInterface;
  public _embeddings: MemoryData|DeviceBuffer;

  constructor(input: UMAPParams) {
    this._umap       = new UMAPBase(input);
    this._embeddings = new DeviceBuffer();
  }

  protected _generate_embeddings(nSamples: number): MemoryData {
    return Series
      .sequence({type: new Float32, size: nSamples * this._umap.nComponents, init: 0, step: 0})
      .data.buffer;
  }

  protected _process_embeddings(embeddings: MemoryData, nSamples: number, returnType: returnType) {
    if (returnType == 'dataframe') {
      return seriesToDataframe(
        Series.new({type: new Float32, data: embeddings}), nSamples, this.nComponents);
    } else if (returnType == 'series') {
      return Series.new({type: new Float32, data: embeddings});
    } else {
      return embeddings;
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
  fitSeries(X: Series<Numeric>, y: null|Series<Numeric>, convertDType: boolean, nFeatures = 1) {
    const nSamples   = Math.floor(X.length / nFeatures);
    this._embeddings = this._generate_embeddings(nSamples);
    let options      = {
      X: X.data.buffer,
      XType: X.type,
      nSamples: nSamples,
      nFeatures: nFeatures,
      convertDType: convertDType,
      embeddings: this._embeddings
    };
    if (y !== null) {
      options = {...options, ...{ y: y.data.buffer, yType: y.type }};
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
  fitDF<T extends Numeric, K extends string>(X: DataFrame<{[P in K]: T}>,
                                             y: null|Series<Numeric>,
                                             convertDType: boolean) {
    const nSamples   = X.numRows;
    const nFeatures  = X.numColumns;
    this._embeddings = this._generate_embeddings(nSamples);

    let options = {
      X: dataframeToSeries(X).data.buffer,
      XType: X.get(X.names[0]).type,
      nSamples: nSamples,
      nFeatures: nFeatures,
      convertDType: convertDType,
      embeddings: this._embeddings
    };
    if (y !== null) {
      options = {...options, ...{ y: y.data.buffer, yType: y.type }};
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
  fit(X: number[], y: number[]|null, convertDType: boolean, nFeatures = 1) {
    const nSamples   = Math.floor(X.length / nFeatures);
    this._embeddings = this._generate_embeddings(nSamples);

    let options = {
      X: X,
      XType: new Float32,
      nSamples: nSamples,
      nFeatures: nFeatures,
      convertDType: convertDType,
      embeddings: this._embeddings
    };
    if (y !== null) {
      options = {...options, ...{ y: y, yType: new Float32 }};
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
  fitTransformSeries<T extends Numeric, R extends returnType>(X: Series<T>,
                                                              y: null|Series<Numeric>,
                                                              convertDType: boolean,
                                                              nFeatures: number,
                                                              returnType: R): returnTypeMap<R, T>;
  fitTransformSeries<T extends Numeric>(X: Series<T>,
                                        y: null|Series<Numeric>,
                                        convertDType: boolean,
                                        nFeatures: number): returnTypeMap<'series', T>;

  fitTransformSeries<T extends Numeric>(X: Series<T>,
                                        y: null|Series<Numeric>,
                                        convertDType: boolean): returnTypeMap<'series', T>;

  fitTransformSeries(X: Series<Numeric>,
                     y: null|Series<Numeric>,
                     convertDType: boolean,
                     nFeatures              = 1,
                     returnType: returnType = 'series') {
    const nSamples = Math.floor(X.length / nFeatures);
    this.fitSeries(X, y, convertDType, nFeatures);
    return this._process_embeddings(this._embeddings, nSamples, returnType);
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
  fitTransformDF<T extends Numeric, K extends string, R extends returnType>(
    X: DataFrame<{[P in K]: T}>, y: null|Series<Numeric>, convertDType: boolean, returnType: R):
    returnTypeMap<R, T>;

  fitTransformDF<T extends Numeric, K extends string>(X: DataFrame<{[P in K]: T}>,
                                                      y: null|Series<Numeric>,
                                                      convertDType: boolean):
    returnTypeMap<'dataframe', T>;

  fitTransformDF<T extends Numeric, K extends string>(X: DataFrame<{[P in K]: T}>,
                                                      y: null|Series<Numeric>,
                                                      convertDType: boolean,
                                                      returnType: returnType = 'dataframe') {
    const nSamples = X.numRows;
    this.fitDF(X, y, convertDType);
    return this._process_embeddings(this._embeddings, nSamples, returnType);
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
  fitTransform(X: number[],
               y: number[]|null,
               convertDType: boolean,
               nFeatures              = 1,
               returnType: returnType = 'series') {
    const nSamples = Math.floor(X.length / nFeatures);
    this.fit(X, y, convertDType, nFeatures);
    return this._process_embeddings(this._embeddings, nSamples, returnType);
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
  transformSeries<T extends Numeric, R extends returnType>(X: Series<T>,
                                                           convertDType: boolean,
                                                           nFeatures: number,
                                                           returnType: R): returnTypeMap<R, T>;

  transformSeries<T extends Numeric>(
    X: Series<T>,
    convertDType: boolean,
    nFeatures: number,
    ): returnTypeMap<'series', T>;

  transformSeries<T extends Numeric>(X: Series<T>,
                                     convertDType: boolean): returnTypeMap<'series', T>;

  transformSeries<T extends Numeric>(X: Series<T>,
                                     convertDType: boolean,
                                     nFeatures              = 1,
                                     returnType: returnType = 'series') {
    const nSamples   = Math.floor(X.length / nFeatures);
    const embeddings = this._generate_embeddings(nSamples);

    const result = this._umap.transform({
      X: X.data.buffer,
      XType: X.type,
      nSamples: nSamples,
      nFeatures: nFeatures,
      convertDType: convertDType,
      embeddings: this._embeddings,
      transformed: embeddings
    });
    return this._process_embeddings(result, nSamples, returnType);
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
  transformDF<T extends Numeric, K extends string, R extends returnType>(
    X: DataFrame<{[P in K]: T}>, convertDType: boolean, returnType: R): returnTypeMap<R, T>;

  transformDF<T extends Numeric, K extends string>(X: DataFrame<{[P in K]: T}>,
                                                   convertDType: boolean):
    returnTypeMap<'dataframe', T>;

  transformDF<T extends Numeric, K extends string>(X: DataFrame<{[P in K]: T}>,
                                                   convertDType: boolean,
                                                   returnType: returnType = 'dataframe') {
    const nSamples   = X.numRows;
    const nFeatures  = X.numColumns;
    const embeddings = this._generate_embeddings(nSamples);

    const result = this._umap.transform({
      X: dataframeToSeries(X).data.buffer,
      XType: X.get(X.names[0]).type,
      nSamples: nSamples,
      nFeatures: nFeatures,
      convertDType: convertDType,
      embeddings: this._embeddings,
      transformed: embeddings
    });
    return this._process_embeddings(result, nSamples, returnType);
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
  transform(X: number[], convertDType: boolean, nFeatures = 1, returnType: returnType = 'series') {
    const nSamples   = Math.floor(X.length / nFeatures);
    const embeddings = this._generate_embeddings(nSamples);

    const result = this._umap.transform({
      X: X,
      XType: new Float32,
      nSamples: nSamples,
      nFeatures: nFeatures,
      convertDType: convertDType,
      embeddings: this._embeddings,
      transformed: embeddings
    });
    return this._process_embeddings(result, nSamples, returnType);
  }

  get embeddings(): Series<Numeric> {
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
