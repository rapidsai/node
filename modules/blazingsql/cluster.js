const cluster = require('cluster');
const { BlazingContext, UcpContext } = require('@rapidsai/blazingsql');
const { Series, DataFrame } = require('@rapidsai/cudf');

const createBlazingContext = 'createBlazingContext';
const createTable = 'createTable';
const runQuery = 'runQuery';
const queryRan = 'queryRan';

const numberOfWorkers = 2;

const ucpContext = new UcpContext();
let bc = null;

if (cluster.isPrimary) {
  cluster.setupMaster({ serialization: 'advanced' });

  const workers = Array(numberOfWorkers);
  for (let i = 0; i < numberOfWorkers; ++i) {
    workers[i] = cluster.fork();
  }

  const ucpMetadata = ["primary", ...Object.keys(cluster.workers)].map((key, idx) => ({
    ralId: idx,
    workerId: key,
    ip: '0.0.0.0',
    port: 4000 + idx,
  }));

  bc = new BlazingContext({
    ralId: 0,
    workerId: "primary",
    ucpMetadata: ucpMetadata.map((xs) => ({ ...xs, ucpContext })),
  });

  workers.forEach((w, idx) => {
    w.send({ operation: createBlazingContext, idx: idx, workerId: w.workerId, ucpMetadata: ucpMetadata })
  });

  const df = createLargeDataFrame();
  const len = Math.ceil(df.numRows / workers.length);
  const table = df.toArrow();
  workers.forEach((w, i) => {
    w.send({
      operation: createTable,
      tableName: 'test_table',
      dataframe: table.slice(i * len, (i + 1) * len).serialize()
    });
  });

  let ctxToken = 0;
  let queryPromises = [];
  workers.forEach((w) => {
    queryPromises.push(new Promise(function (resolve) {
      w.send({ operation: runQuery, ctxToken: ctxToken++, query: 'SELECT a FROM test_table' });
      w.on('message', (args) => {
        console.log(`Finished query on token: ${args.ctxToken}`);
        resolve(args);
      });
    }));
  });

  Promise.all(queryPromises).then(function (results) {
    console.log('Finished running all queries.');
    results.forEach((result) => {
      console.log(DataFrame.fromArrow(result.dataframe));
    });
    workers.forEach((w) => w.kill());
  });

} else if (cluster.isWorker) {
  process.on('message', (args) => {
    if (args.operation === createBlazingContext) {
      bc = new BlazingContext({
        ralId: args.idx,
        workerId: args.workerId,
        ucpMetadata: args.ucpMetadata.map((xs) => ({ ...xs, ucpContext })),
      });
    }

    if (args.operation === createTable) {
      console.log(`Creating table: ${args.tableName}`);
      bc.createTable(args.tableName, DataFrame.fromArrow(args.dataframe));
    }

    if (args.operation === runQuery) {
      console.log(`Token: ${args.ctxToken}`);
      const result = bc.sql(args.query, args.ctxToken);
      process.send({ operation: queryRan, ctxToken: args.ctxToken, dataframe: result.toArrow().serialize() });
    }
  });
}

function createLargeDataFrame() {
  const a = Series.new(Array.from(Array(300).keys()));
  return new DataFrame({ 'a': a, 'b': a });
}
