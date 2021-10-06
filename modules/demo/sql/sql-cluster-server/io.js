const {SQLCluster} = require('@rapidsai/sql');

async function test() {
  const bc = await SQLCluster.init({numWorkers: 2});
  await bc.createTable('test_table', [`${__dirname}/data/wiki_page_0.csv`]);

  const result = await bc.sql('SELECT * FROM test_table');

  console.log(result.toString());

  bc.kill();
}
test();
