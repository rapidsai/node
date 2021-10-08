const {SQLCluster}        = require('@rapidsai/sql');
const {Series, DataFrame} = require('@rapidsai/cudf');

async function test() {
  const a  = Series.new([1, 2, 3]);
  const df = new DataFrame({'a': a});

  const bc = await SQLCluster.init({port: 10000});
  await bc.createTable('test_table', df);

  const result = await bc.sql('SELECT * FROM test_table');

  result.forEach((r) => { console.log(r.toString()); });

  bc.kill();
}
test();
