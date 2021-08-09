const { BlazingCluster } = require('@rapidsai/blazingsql');
const { Series, DataFrame } = require('@rapidsai/cudf');

const df = createLargeDataFrame();

const bc = new BlazingCluster({ numWorkers: 1 });
bc.createTable('test_table', df);

bc.sql('SELECT a FROM test_table');

function createLargeDataFrame() {
  const a = Series.new(Array.from(Array(300).keys()));
  return new DataFrame({ 'a': a, 'b': a });
}
