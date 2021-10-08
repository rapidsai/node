const {Series, DataFrame} = require('@rapidsai/cudf');
const {SQLCluster}        = require('@rapidsai/sql');

async function test() {
  const a  = Series.new([1, 2, 3]);
  const df = new DataFrame({'a': a, 'b': a, 'c': a, 'd': a, 'e': a});

  const bc = await SQLCluster.init({port: 10000, numWorkers: 2});
  await bc.createTable('test_table', df);

  const result = await bc.sql('SELECT * FROM test_table WHERE a > 1');

  result.forEach((df) => { console.log(df.toString()); });

  bc.kill();
}
test();
