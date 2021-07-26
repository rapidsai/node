const cluster = require('cluster');
const { BlazingContext, UcpContext } = require('@rapidsai/blazingsql');
const { Series, DataFrame } = require('@rapidsai/cudf');

const createTable = 'createTable';
const runQuery = 'runQuery';
const queryRan = 'queryRan';

const numberOfWorkers = 2;

const ucp_context = new UcpContext();
const queryToRun = 'SELECT a FROM test_table';

if (cluster.isMaster) {
  cluster.setupMaster({ serialization: 'advanced' });

  const workers = Array(numberOfWorkers);
  for (let i = 0; i < numberOfWorkers; ++i) {
    workers[i] = cluster.fork();
  }

  const bc = new BlazingContext(Object.keys(cluster.workers).map((id) => {
    return {
      workerId: id,
      ip: 'localhost',
      port: 8000,
      ucpContext: ucp_context,
    };
  }));

  const a = Series.new([1, 2, 3]);
  const b = Series.new([1, 2, 3]);
  const df = new DataFrame({ 'a': a, 'b': b });

  workers.forEach((w) => {
    w.send({ operation: createTable, tableName: 'test_table', dataframe: df.toArrow().serialize() });
  });

  let ctxToken = 0;
  let queryPromises = [];
  workers.forEach((w) => {
    queryPromises.push(new Promise(function (resolve) {
      w.send({ operation: runQuery, ctxToken: ctxToken++, query: queryToRun });
      w.on('message', (args) => {
        console.log(`Finished query on token: ${args.ctxToken}`);
        resolve(args);
      });
    }));
  });

  Promise.all(queryPromises).then(function (results) {
    console.log('Finished running all queries.');
    results.forEach((result) => {
      console.log(DataFrame.fromArrow(result.df));
    });
    workers.forEach((w) => w.kill());
  });

} else if (cluster.isWorker) {

  const bc = new BlazingContext([
    {
      workerId: cluster.worker,
      ip: 'localhost',
      port: 8000,
      ucpContext: ucp_context,
    }
  ]);

  process.on('message', (args) => {
    if (args.operation === createTable) {
      console.log(`Creating table: ${args.tableName}`);
      bc.createTable(args.tableName, DataFrame.fromArrow(args.dataframe));
    }

    if (args.operation === runQuery) {
      console.log(`Token: ${args.ctxToken}`);
      const result = bc.sql(args.query, args.ctxToken);
      process.send({ operation: queryRan, ctxToken: args.ctxToken, df: result.toArrow().serialize() });
    }
  });
}
