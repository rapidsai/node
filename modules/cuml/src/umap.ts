import {
  DataFrame,
  Float32,
  Float64,
  Int32,
  Int64,
  Int8,
  Series,
  TypeMap,
  Uint32,
} from '@rapidsai/cudf';
import {DeviceBuffer} from '@rapidsai/rmm/';

import {UMAPBase, UMAPInterface, UMAPParams} from './umap_base';
import {series_to_dataframe, transform_input_to_device_buffer} from './utilities/array_utils';

export type Numeric = Float32|Float64|Int32|Int64|Int8|Uint32;

export class UMAP {
  public _umap: UMAPInterface;
  public _embeddings: DeviceBuffer;

  protected constructor(input: UMAPParams) {
    this._umap       = new UMAPBase(input);
    this._embeddings = new DeviceBuffer();
  }

  fit(X: Series<Numeric>|DataFrame<TypeMap>,
      y: null|Series<Numeric>|DataFrame<TypeMap>,
      convertDType: boolean): void {
    const n_samples  = (X instanceof Series) ? X.length : X.numRows;
    const n_features = (X instanceof Series) ? 1 : X.numColumns;
    this._embeddings = transform_input_to_device_buffer(Series.sequence(
      {type: new Float32, size: n_samples * this._umap.nComponents, init: 0, step: 0}));

    this._embeddings = this._umap.fit(transform_input_to_device_buffer(X),
                                      n_samples,
                                      n_features,
                                      (y == null) ? null : transform_input_to_device_buffer(y),
                                      null,
                                      null,
                                      convertDType,
                                      this._embeddings);
    return undefined;
  }

  fit_transform(X: Series<Numeric>|DataFrame<TypeMap>,
                y: null|Series<Numeric>|DataFrame<TypeMap>,
                convertDType: boolean): DataFrame {
    const n_samples = (X instanceof Series) ? X.length : X.numRows;
    this.fit(X, y, convertDType);
    return series_to_dataframe(this.embeddings, n_samples, this.nComponents);
  }

  transform(X: Series<Numeric>|DataFrame<TypeMap>, convertDType: boolean): DataFrame {
    const n_samples  = (X instanceof Series) ? X.length : X.numRows;
    const n_features = (X instanceof Series) ? 1 : X.numColumns;
    const embeddings = transform_input_to_device_buffer(Series.sequence(
      {type: new Float32, size: n_samples * this._umap.nComponents, init: 0, step: 0}));

    const result = this._umap.transform(transform_input_to_device_buffer(X),
                                        n_samples,
                                        n_features,
                                        null,
                                        null,
                                        convertDType,
                                        this._embeddings,
                                        embeddings);
    return series_to_dataframe(
      Series.new({type: new Float32, data: result}), n_samples, this.nComponents);
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
}
