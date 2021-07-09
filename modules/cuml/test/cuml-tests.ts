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
  sources: [__dirname + './iris.csv'],
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

test('fit trustworthiness score (series)', () => {
  const umap   = new UMAP(options);
  const t1     = umap.fitSeries(XSeries, y, 4).embeddings;
  const trust1 = trustworthinessSeries(XSeries, t1.asSeries(), 4);

  expect(trust1).toBeGreaterThan(0.97);
});

test('fit trustworthiness score (dataframe)', () => {
  const umap   = new UMAP(options);
  const t1     = umap.fitDataFrame(X, y).embeddings;
  const trust1 = trustworthinessDataFrame(X, t1.asDataFrame());

  expect(trust1).toBeGreaterThan(0.97);
});

test('fit trustworthiness score (array)', () => {
  const umap   = new UMAP(options);
  const t1     = umap.fitArray([...XSeries], [...y], 4).embeddings;
  const trust1 = trustworthiness([...XSeries], [...t1.asSeries()], 4);

  expect(trust1).toBeGreaterThan(0.97);
});

test('transform trustworthiness score (series)', () => {
  const umap  = new UMAP(options);
  const t1    = umap.fitSeries(XSeries, y, 4).transformSeries(XSeries, 4);
  const score = trustworthinessSeries(XSeries, t1.asSeries(), 4);

  expect(score).toBeGreaterThan(0.95);
});

test('transform trustworthiness score (dataframe)', () => {
  const umap  = new UMAP(options);
  const t1    = umap.fitDataFrame(X, y).transformDataFrame(X);
  const score = trustworthinessDataFrame(X, t1.asDataFrame());

  expect(score).toBeGreaterThan(0.95);
});

test('transform trustworthiness score (array)', () => {
  const umap  = new UMAP(options);
  const t1    = umap.fitArray([...XSeries], [...y], 4).transformArray([...XSeries], 4);
  const score = trustworthiness([...XSeries], [...t1.asSeries()], 4);

  expect(score).toBeGreaterThan(0.95);
});
