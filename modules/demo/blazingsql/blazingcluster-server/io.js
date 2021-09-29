const {SQLContext}        = require('@rapidsai/blazingsql');
const {Series, DataFrame} = require('@rapidsai/cudf');

const a = Series.new(['test']);

const bc = new SQLContext();
bc.createTable('test_table', [`${__dirname}/wikipedia_pages.csv`]);
bc.createTable(
  'test_table2',
  new DataFrame({'a': a}),
)

const result = bc.sql('SELECT * FROM test_table').result();

console.log(result.names);
console.log(bc.describeTable('test_table'));

result.names.forEach((n) => { console.log([...result.get(n)]); });
