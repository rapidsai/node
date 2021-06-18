import {DataFrame} from '@rapidsai/cudf';
import {
  trustworthiness,
  trustworthinessDataFrame,
  trustworthinessSeries,
  UMAP
} from '@rapidsai/cuml';

const df = DataFrame.readCSV({
  header: 0,
  sourceType: 'files',
  sources: ['./debug/iris.csv'],
  dataTypes: {
    sepal_length: 'float32',
    sepal_width: 'float32',
    petal_length: 'float32',
    petal_width: 'float32',
    target: 'float32'
  }
});
const X  = df.drop(['target']);
const y  = df.get('target');

const XSeries = X.interleaveColumns();
const options = {
  nNeighbors: 10,
  minDist: 0.01,
  randomState: 12,
  targetNNeighbors: -1
};

test('fit_transform trustworthiness score (series)', () => {
  const umap   = new UMAP(options);
  const t1     = umap.fitTransformSeries(XSeries, y, false, 4).asSeries();
  const trust1 = trustworthinessSeries(XSeries, t1, 4);

  expect(trust1).toBeGreaterThan(0.97);
});

test('fit_transform trustworthiness score (dataframe)', () => {
  const umap   = new UMAP(options);
  const t1     = umap.fitTransformDataFrame(X, y, true).asSeries();
  const trust1 = trustworthinessSeries(XSeries, t1, 4);

  expect(trust1).toBeGreaterThan(0.97);
});

test('fit_transform trustworthiness score (array)', () => {
  const umap   = new UMAP(options);
  const t1     = umap.fitTransform([...XSeries], [...y], true, 4).asSeries();
  const trust1 = trustworthiness([...XSeries], [...t1], 4);

  expect(trust1).toBeGreaterThan(0.97);
});

test('transform trustworthiness score (series)', () => {
  const umap  = new UMAP(options);
  const t1    = umap.fitSeries(XSeries, y, true, 4).transform(true).asSeries();
  const score = trustworthinessSeries(XSeries, t1, 4);

  expect(score).toBeGreaterThan(0.95);
});

test('transform trustworthiness score (dataframe)', () => {
  const umap  = new UMAP(options);
  const t1    = umap.fitDataFrame(X, y, true).transform(true).asDataFrame();
  const score = trustworthinessDataFrame(X, t1);

  expect(score).toBeGreaterThan(0.95);
});

test('transform trustworthiness score (array)', () => {
  const umap  = new UMAP(options);
  const t1    = umap.fit([...XSeries], [...y], true, 4).transform(true).asSeries();
  const score = trustworthiness([...XSeries], [...t1], 4);

  expect(score).toBeGreaterThan(0.95);
});
