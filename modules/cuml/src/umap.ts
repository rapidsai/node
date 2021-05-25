import {
  DataFrame,
  Float32,
  Float64,
  Int32,
  Int64,
  Int8,
  // DataType,
  // Float64Series,
  Series,
  TypeMap,
  Uint32,
} from '@rapidsai/cudf';
import {DeviceBuffer} from '@rapidsai/rmm/';

import {UMAPBase, UMAPInterface, UMAPParams} from './umap_base';

export type Numeric = Float32|Float64|Int32|Int64|Int8|Uint32;

export class UMAP {
  public _umap: UMAPInterface;

  protected constructor(input: UMAPParams) { this._umap = new UMAPBase(input); }

  /**
   * convert a dataframe to a single series to replicate conversion to a matrix in an organized
   * for {x1,y1,x2,y2...} for df = {x:[x1,x2], y: [y1,y2]}
   */
  private _dataframe_to_series(input: DataFrame): Series<Float32> {
    let result = Series.sequence(
      {type: new Float32, init: 0, size: input.numRows * input.numColumns, step: 0});
    let _i = 0;
    input.names.forEach((col) => {
      result = result.scatter(
        input.get(col),
        Series.sequence({type: new Int32, init: _i, size: input.numRows, step: input.numColumns}));
      _i++;
    });
    return result;
  }

  private _series_to_dataframe(input: DeviceBuffer, n_samples: number, n_features: number):
    DataFrame {
    const input_series = Series.new({type: new Float32, data: input});
    let result         = new DataFrame({});
    for (let i = 0; i < n_features; i++) {
      result = result.assign({
        [i]: input_series.gather(
          Series.sequence({type: new Int32, init: i * n_samples, size: n_samples, step: 1}))
      });
    }
    return result;
  }

  /**
   * convert input of type Series or DataFrame to a DeviceBuffer
   * @param input
   * @returns
   */
  private _transform_input_to_device_buffer(input: Series<Numeric>|
                                            DataFrame<TypeMap>): DeviceBuffer {
    if (input instanceof DataFrame) {
      return new DeviceBuffer(this._dataframe_to_series(input).data.buffer);
    }
    return new DeviceBuffer(input.data.buffer);
  }

  fit(X: Series<Numeric>|DataFrame<TypeMap>,
      y: null|Series<Numeric>|DataFrame<TypeMap>,
      convertDType: boolean): void {
    const n_samples  = (X instanceof Series) ? X.length : X.numRows;
    const n_features = (X instanceof Series) ? 1 : X.numColumns;

    this._umap.fit(this._transform_input_to_device_buffer(X),
                   n_samples,
                   n_features,
                   (y == null) ? null : this._transform_input_to_device_buffer(y),
                   null,
                   null,
                   convertDType);
  }

  transform(X: Series<Numeric>|DataFrame<TypeMap>, convertDType: boolean): DataFrame {
    const n_samples  = (X instanceof Series) ? X.length : X.numRows;
    const n_features = (X instanceof Series) ? 1 : X.numColumns;

    const result = this._umap.transform(
      this._transform_input_to_device_buffer(X), n_samples, n_features, null, null, convertDType);

    return this._series_to_dataframe(result, n_samples, n_features);
  }

  get embeddings(): DeviceBuffer { return this._umap.getEmbeddings(); }
}
