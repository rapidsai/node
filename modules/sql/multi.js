const {SQLCluster}        = require('@rapidsai/sql');
const {Series, DataFrame} = require('@rapidsai/cudf');

async function test() {
  const a  = Series.new([1, 2, 3]);
  const df = new DataFrame({'a': a});

  const bc = await SQLCluster.init();
  await bc.createTable('test_table', df);

  const result = await bc.sql('SELECT * FROM test_table');

  console.log(result.toString());

  bc.kill();
}
test();
