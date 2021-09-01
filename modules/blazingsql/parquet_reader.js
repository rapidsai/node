const { DataFrame } = require('@rapidsai/cudf');

const df = DataFrame.readParquet({
  sources: [`${__dirname}/parquettest`],
});

console.log(df.names);
