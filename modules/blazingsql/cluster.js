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

  const ucpMetadata = ['0', ...Object.keys(cluster.workers)].map((key, idx) => ({
    ralId: idx,
    workerId: key,
    ip: '0.0.0.0',
    port: 4000 + idx,
  }));

  bc = new BlazingContext({
    ralId: 0,
    workerId: '0',
    enableLogging: true,
    // networkIfaceName: 'eno1',
    workersUcpInfo: ucpMetadata.map((xs) => ({ ...xs, ucpContext }))
  });

  workers.forEach((w, idx) => {
    console.log({ workerId: w.id });
    w.send({ operation: createBlazingContext, idx: idx + 1, workerId: w.id, ucpMetadata: ucpMetadata })
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
      ctxToken = ctxToken + 1;
      w.send({ operation: runQuery, ctxToken: ctxToken, messageId: `message_${ctxToken.toString()}`, query: 'SELECT a FROM test_table' });
      w.on('message', (args) => {
        console.log(`Finished query on token: ${ctxToken}`);
        resolve(args);
      });
    }));
  });

  Promise.all(queryPromises).then(function (results) {
    console.log('Finished running all queries.');
    results.forEach((result) => {
      console.log(`pulling result with messageId=${result.messageId}`);
      bc.pullFromCache(result.messageId);
    });
    workers.forEach((w) => w.kill());
  });

} else if (cluster.isWorker) {
  process.on('message', (args) => {
    const { operation, ...rest } = args;
    const {
      ctxToken, dataframe, idx, messageId, query, tableName, ucpMetadata, workerId
    } = rest;
    console.log(`message "${operation}":`, rest);
    if (operation === createBlazingContext) {
      bc = new BlazingContext({
        ralId: idx,
        workerId: workerId,
        enableLogging: true,
        // networkIfaceName: 'eno1',
        workersUcpInfo: ucpMetadata.map((xs) => ({ ...xs, ucpContext })),
      });
    }

    if (operation === createTable) {
      console.log(`Creating table: ${tableName}`);
      bc.createTable(tableName, DataFrame.fromArrow(dataframe));
    }

    if (operation === runQuery) {
      console.log(`Token: ${ctxToken}`);
      const result = bc.sql(query, ctxToken);
      result.sendTo(0, messageId);
      process.send({ operation: queryRan, ctxToken: ctxToken, messageId: messageId });
    }
  });
}

function createLargeDataFrame() {
  const a = Series.new(Array.from(Array(300).keys()));
  return new DataFrame({ 'a': a, 'b': a });
}
