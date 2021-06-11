import {MemoryData} from '@nvidia/cuda';
import {DataFrame, Float32, Numeric, Series} from '@rapidsai/cudf';
import {DeviceBuffer} from '@rapidsai/rmm';

import {CUMLLogLevels, MetricType} from './mappings';
import {UMAPBase, UMAPInterface, UMAPParams} from './umap_base';
import {dataframe_to_series, series_to_dataframe} from './utilities/array_utils';

type returnType = 'dataframe'|'series'|'devicebuffer';

export type returnTypeMap<T extends returnType, R extends Numeric> = {
  'dataframe': DataFrame<{[P in number]: R}>,
  'series': Series<R>,
  'devicebuffer': MemoryData
}[T];

export class UMAP {
  public _umap: UMAPInterface;
  public _embeddings: MemoryData|DeviceBuffer;

  constructor(input: UMAPParams) {
    this._umap       = new UMAPBase(input);
    this._embeddings = new DeviceBuffer();
  }

  protected _generate_embeddings(n_samples: number): MemoryData {
    return Series
      .sequence({type: new Float32, size: n_samples * this._umap.nComponents, init: 0, step: 0})
      .data.buffer;
  }

  protected _process_embeddings(embeddings: MemoryData, n_samples: number, returnType: returnType) {
    if (returnType == 'dataframe') {
      return series_to_dataframe(
        Series.new({type: new Float32, data: embeddings}), n_samples, this.nComponents);
    } else if (returnType == 'series') {
      return Series.new({type: new Float32, data: embeddings});
    } else {
      return embeddings;
    }
  }
  /**
   * Fit X into an embedded space
   * @param X Dense or sparse matrix containing floats or doubles. Acceptable dense formats: cuDF
   *   DataFrame/Series
   * @param y cuDF Series containing target values
   * @param convertDType When set to True, the method will automatically convert the inputs to
   *   float32
   */
  fitSeries(X: Series<Numeric>, y: null|Series<Numeric>, convertDType: boolean) {
    const n_samples  = X.length;
    const n_features = 1;
    this._embeddings = this._generate_embeddings(n_samples);

    this._embeddings = this._umap.fit(X.data.buffer,
                                      n_samples,
                                      n_features,
                                      (y == null) ? null : y.data.buffer,
                                      null,
                                      null,
                                      convertDType,
                                      this._embeddings);
  }

  fitDF<T extends Numeric, K extends string>(X: DataFrame<{[P in K]: T}>,
                                             y: null|Series<Numeric>,
                                             convertDType: boolean) {
    const n_samples  = X.numRows;
    const n_features = X.numColumns;
    this._embeddings = this._generate_embeddings(n_samples);

    this._embeddings = this._umap.fit(dataframe_to_series(X).data.buffer,
                                      n_samples,
                                      n_features,
                                      (y == null) ? null : y.data.buffer,
                                      null,
                                      null,
                                      convertDType,
                                      this._embeddings);
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
   *   DataFrame/Series
   * @param y cuDF Series containing target values
   * @param convertDType When set to True, the method will automatically convert the inputs to
   *   float32
   * @param returnType Desired output type of results and attributes of the estimators
   *
   * @returns Embedding of the data in low-dimensional space
   */
  fitTransformSeries<T extends Numeric, R extends returnType>(X: Series<T>,
                                                              y: null|Series<Numeric>,
                                                              convertDType: boolean,
                                                              returnType: R): returnTypeMap<R, T>;
  fitTransformSeries<T extends Numeric>(
    X: Series<T>,
    y: null|Series<Numeric>,
    convertDType: boolean,
    ): returnTypeMap<'dataframe', T>;

  fitTransformSeries(X: Series<Numeric>,
                     y: null|Series<Numeric>,
                     convertDType: boolean,
                     returnType: returnType = 'dataframe') {
    const n_samples = X.length;
    this.fitSeries(X, y, convertDType);
    return this._process_embeddings(this._embeddings, n_samples, returnType);
  }

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
    const n_samples = X.numRows;
    this.fitDF(X, y, convertDType);
    return this._process_embeddings(this._embeddings, n_samples, returnType);
  }

  /**
   * Transform X into the existing embedded space and return that transformed output.
   *
   * @param X Dense or sparse matrix containing floats or doubles. Acceptable dense formats: cuDF
   *   DataFrame/Series
   * @param convertDType When set to True, the method will automatically convert the inputs to
   *   float32
   * @param returnType Desired output type of results and attributes of the estimators
   *
   * @returns Embedding of the data in low-dimensional space
   */
  transformSeries<T extends Numeric, R extends returnType>(X: Series<T>,
                                                           convertDType: boolean,
                                                           returnType: R): returnTypeMap<R, T>;

  transformSeries<T extends Numeric>(X: Series<T>,
                                     convertDType: boolean): returnTypeMap<'dataframe', T>;

  transformSeries<T extends Numeric>(X: Series<T>,
                                     convertDType: boolean,
                                     returnType: returnType = 'dataframe') {
    const n_samples  = X.length;
    const n_features = 1;
    const embeddings = this._generate_embeddings(n_samples);

    const result = this._umap.transform(
      X.data.buffer, n_samples, n_features, null, null, convertDType, this._embeddings, embeddings);
    return this._process_embeddings(result, n_samples, returnType);
  }

  transformDF<T extends Numeric, K extends string, R extends returnType>(
    X: DataFrame<{[P in K]: T}>, convertDType: boolean, returnType: R): returnTypeMap<R, T>;

  transformDF<T extends Numeric, K extends string>(X: DataFrame<{[P in K]: T}>,
                                                   convertDType: boolean):
    returnTypeMap<'dataframe', T>;

  transformDF(X: any, convertDType: boolean, returnType: returnType = 'dataframe') {
    const n_samples  = X.numRows;
    const n_features = X.numColumns;
    const embeddings = this._generate_embeddings(n_samples);

    const result = this._umap.transform(dataframe_to_series(X).data.buffer,
                                        n_samples,
                                        n_features,
                                        null,
                                        null,
                                        convertDType,
                                        this._embeddings,
                                        embeddings);
    return this._process_embeddings(result, n_samples, returnType);
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
