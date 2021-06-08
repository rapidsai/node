const { UMAP, trustworthiness, MetricType, CUMLLogLevels } = require('@rapidsai/cuml');
const { Series, DataFrame, Float64, Int32, Float32 } = require('@rapidsai/cudf');
const { DeviceBuffer } = require('@rapidsai/rmm');

const df = DataFrame.readCSV({
  header: 0,
  sourceType: 'files',
  sources: ['../cuml/debug/iris.csv'],
  dataTypes: {
    sepal_length: 'float32',
    sepal_width: 'float32',
    petal_length: 'float32',
    petal_width: 'float32',
    target: 'int32'
  }
});

console.log("n_samples", df.numRows);
console.log("n_features", df.numColumns - 1);
const y = df.get('target')

umap = new UMAP({ nNeighbors: 10, minDist: 0.01, randomState: 12, targetNNeighbors: -1, targetMetric: MetricType.CATEGORICAL, verbosity: CUMLLogLevels.CUML_LEVEL_CRITICAL });

t1 = umap.fit_transform(df.select(['sepal_length', 'sepal_width', 'petal_length', 'petal_width']), y, true);
console.log("fit_transform");

tm = trustworthiness(df.select(['sepal_length', 'sepal_width', 'petal_length', 'petal_width']), t1, 10);
console.log(tm); // should be greater than 0.97

umap = new UMAP({
  nNeighbors: 10,
  minDist: 0.01,
  nEpochs: 800,
  init: 0,
  randomState: 42,
  targetNNeighbors: -1,
  targetMetric: MetricType.CATEGORICAL
});

console.log("fit & transform");

umap.fit(df.drop(['target']), y, true);
t = umap.transform(df.drop(['target']), y, true);

tm = trustworthiness(df.drop(['target']), t, 10);
console.log(tm); // should be greater than 0.97
