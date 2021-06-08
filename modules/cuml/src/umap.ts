import {
  DataFrame,
  Float32,
  Numeric,
  Series,
  TypeMap,
} from '@rapidsai/cudf';
import {DeviceBuffer} from '@rapidsai/rmm/';
import {CUMLLogLevels, MetricType} from './mappings';

import {UMAPBase, UMAPInterface, UMAPParams} from './umap_base';
import {series_to_dataframe, transform_input_to_device_buffer} from './utilities/array_utils';

export class UMAP {
  public _umap: UMAPInterface;
  public _embeddings: DeviceBuffer;

  protected constructor(input: UMAPParams) {
    this._umap       = new UMAPBase(input);
    this._embeddings = new DeviceBuffer();
  }

  private _generate_embeddings(n_samples: number): Series<Float32> {
    return Series.sequence(
      {type: new Float32, size: n_samples * this._umap.nComponents, init: 0, step: 0});
  }

  fit(X: Series<Numeric>|DataFrame<TypeMap>,
      y: null|Series<Numeric>|DataFrame<TypeMap>,
      convertDType: boolean): void {
    const n_samples  = (X instanceof Series) ? X.length : X.numRows;
    const n_features = (X instanceof Series) ? 1 : X.numColumns;
    this._embeddings = transform_input_to_device_buffer(this._generate_embeddings(n_samples));

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

  _process_embeddings(embeddings: DeviceBuffer,
                      n_samples: number,
                      returnType: 'dataframe'|'series'|'devicebuffer') {
    if (returnType == 'dataframe') {
      return series_to_dataframe(embeddings, n_samples, this.nComponents);
    } else if (returnType == 'series') {
      return Series.new({type: new Float32, data: embeddings});
    } else {
      return embeddings;
    }
  }
  fit_transform(X: Series<Numeric>|DataFrame<TypeMap>,
                y: null|Series<Numeric>|DataFrame<TypeMap>,
                convertDType: boolean,
                returnType: 'dataframe'|'series'|'devicebuffer' = 'dataframe'): DataFrame
    |Series<Float32>|DeviceBuffer {
    const n_samples = (X instanceof Series) ? X.length : X.numRows;
    this.fit(X, y, convertDType);
    return this._process_embeddings(this._embeddings, n_samples, returnType);
  }

  transform(X: Series<Numeric>|DataFrame<TypeMap>,
            convertDType: boolean,
            returnType: 'dataframe'|'series'|'devicebuffer' = 'dataframe'): DataFrame
    |Series<Float32>|DeviceBuffer {
    const n_samples  = (X instanceof Series) ? X.length : X.numRows;
    const n_features = (X instanceof Series) ? 1 : X.numColumns;
    const embeddings = transform_input_to_device_buffer(this._generate_embeddings(n_samples));

    const result = this._umap.transform(transform_input_to_device_buffer(X),
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
