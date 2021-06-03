import {DataFrame, Series, TypeMap} from '@rapidsai/cudf';

import {CUML} from './addon';
import {Numeric, transform_input_to_device_buffer} from './utilities/array_utils';

export function trustworthiness(X: Series<Numeric>|DataFrame<TypeMap>,
                                embedded: Series<Numeric>|DataFrame<TypeMap>,
                                n_neighbors = 5,
                                batch_size  = 10) {
  const n_samples  = (X instanceof Series) ? X.length : X.numRows;
  const n_features = (X instanceof Series) ? 1 : X.numColumns;

  const n_components = (embedded instanceof Series) ? 1 : embedded.numColumns;

  return CUML.trustworthiness(transform_input_to_device_buffer(X),
                              transform_input_to_device_buffer(embedded),
                              n_samples,
                              n_features,
                              n_components,
                              n_neighbors,
                              batch_size);
}
