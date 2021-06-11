import {
  DataFrame,
  Int32,
  Numeric,
  Series,
} from '@rapidsai/cudf';

/**
 * convert a dataframe to a single series to replicate conversion to a matrix in an organized
 * for {x1,y1,x2,y2...} for df = {x:[x1,x2], y: [y1,y2]}
 */
export function dataframe_to_series<T extends Numeric, K extends string>(
  input: DataFrame<{[P in K]: T}>): Series<Numeric> {
  return input.interleaveColumns();
}

/**
 * convert a series to a dataframe as per the number of componenet in umapparams
 * @param input
 * @param n_samples
 * @param nComponents
 * @returns DataFrame
 */
export function series_to_dataframe<T extends Numeric>(
  input: Series<T>, nSamples: number, nComponents: number) {
  let result = new DataFrame<{[P in number]: T}>({});
  for (let i = 0; i < nComponents; i++) {
    result = result.assign({
      [i]:
        input.gather(Series.sequence({type: new Int32, init: i, size: nSamples, step: nComponents}))
    });
  }
  return result;
}
