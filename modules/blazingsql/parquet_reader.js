const { DataFrame } = require('@rapidsai/cudf');

const df = DataFrame.readParquet({
  columns: ['a'],
  sources: [`${__dirname}/parquettest`],
});

console.log(df.names);
