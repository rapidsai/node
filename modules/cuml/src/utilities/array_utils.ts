import {
  DataFrame,
  Int32,
  Numeric,
  Series,
  TypeMap,
} from '@rapidsai/cudf';
import {DeviceBuffer} from '@rapidsai/rmm/';

/**
 * convert a dataframe to a single series to replicate conversion to a matrix in an organized
 * for {x1,y1,x2,y2...} for df = {x:[x1,x2], y: [y1,y2]}
 */
export function dataframe_to_series(input: DataFrame): Series<Numeric> {
  return input.interleaveColumns();
}

/**
 * convert input of type Series or DataFrame to a DeviceBuffer
 * @param input
 * @returns
 */
export function transform_input_to_device_buffer(input: Series|DataFrame<TypeMap>): DeviceBuffer {
  if (input instanceof DataFrame) { return new DeviceBuffer(dataframe_to_series(input).data); }
  return new DeviceBuffer(input.data);
}

export function series_to_dataframe(
  input: Series, n_samples: number, nComponents: number): DataFrame {
  let result = new DataFrame({});
  for (let i = 0; i < nComponents; i++) {
    result = result.assign({
      [i]: input.gather(
        Series.sequence({type: new Int32, init: i, size: n_samples, step: nComponents}))
    });
  }
  return result;
}
