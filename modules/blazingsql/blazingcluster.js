const { BlazingCluster } = require('@rapidsai/blazingsql');
const { Series, DataFrame } = require('@rapidsai/cudf');

const a = Series.new([1, 2, 3]);
const df = new DataFrame({ 'a': a });

const bc = new BlazingCluster({ numWorkers: 1 });
