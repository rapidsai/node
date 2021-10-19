const {Series, DataFrame} = require('@rapidsai/cudf');
const {SQLContext}        = require('@rapidsai/sql');

async function test() {
  const bc = new SQLContext();
  bc.createParquetTable('test_table', [`${__dirname}/test.parquet`]);

  const results = await bc.sql('SELECT * FROM test_table').result();
  results.forEach((result) => { console.log(result.toString()); });
}
test();
