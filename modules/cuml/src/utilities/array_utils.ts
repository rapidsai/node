import {
  DataFrame,
  Float32,
  Float64,
  Int32,
  Int64,
  Int8,
  Series,
  TypeMap,
  Uint32
} from '@rapidsai/cudf';
import {DeviceBuffer} from '@rapidsai/rmm/';
export type Numeric = Float32|Float64|Int32|Int64|Int8|Uint32;

/**
 * convert a dataframe to a single series to replicate conversion to a matrix in an organized
 * for {x1,y1,x2,y2...} for df = {x:[x1,x2], y: [y1,y2]}
 */
export function dataframe_to_series(input: DataFrame): Series<Numeric> {
  const type = input.get(input.names[0]).type;
  let result = Series.sequence(
    {type: new type.constructor, init: 0, size: input.numRows * input.numColumns, step: 0});
  let _i = 0;
  input.names.forEach((col) => {
    if (!type.compareTo(input.get(col).type)) {
      throw new Error('All columns must have same type');
    }
    result = result.scatter(
      input.get(col),
      // Series.sequence({type: new Int32, init: _i * input.numRows, size: input.numRows, step:
      // 1}));
      Series.sequence({type: new Int32, init: _i, size: input.numRows, step: input.numColumns}));
    _i++;
  });
  return result;
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
  // console.log([...input], input.length);
  let result = new DataFrame({});
  for (let i = 0; i < nComponents; i++) {
    result = result.assign({
      [i]: input.gather(
        Series.sequence({type: new Int32, init: i, size: n_samples, step: nComponents}))
      // {type: new Int32, init: i * n_samples, size: n_samples, step: 1}))
    });
  }
  return result;
}
