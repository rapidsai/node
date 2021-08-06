const cluster = require('cluster');
const { BlazingContext, UcpContext } = require('@rapidsai/blazingsql');
const { Series, DataFrame } = require('@rapidsai/cudf');

const createBlazingContext = 'createBlazingContext';
const createTable = 'createTable';
const runQuery = 'runQuery';
const queryRan = 'queryRan';

const numberOfWorkers = 2;
const configOptions = {
  PROTOCOL: 'UCX',
  ENABLE_TASK_LOGS: true,
  ENABLE_COMMS_LOGS: true,
  ENABLE_OTHER_ENGINE_LOGS: true,
  ENABLE_GENERAL_ENGINE_LOGS: true,
  LOGGING_FLUSH_LEVEL: 'trace',
  BLAZING_CACHE_DIRECTORY: '/tmp',
  BLAZING_LOGGING_DIRECTORY: `${__dirname}/z-log`,
  BLAZING_LOCAL_LOGGING_DIRECTORY: `${__dirname}/z-log`,
};

const ucpContext = new UcpContext();
let bc = null;

if (cluster.isPrimary) {

  const fs = require('fs');
  fs.rmSync(`${__dirname}/z-log`, { force: true, recursive: true });
  fs.mkdirSync(`${__dirname}/z-log`);

  cluster.setupMaster({ serialization: 'advanced' });

  const workers = Array(numberOfWorkers);
  for (let i = 0; i < numberOfWorkers; ++i) {
    workers[i] = cluster.fork();
  }

  const ucpMetadata = ['0', ...Object.keys(cluster.workers)].map((key, idx) => ({
    workerId: key,
    ip: '0.0.0.0',
    port: 4000 + idx,
  }));

  workers.forEach((w) => w.send({ operation: createBlazingContext, ucpMetadata }));

  bc = createContext(0, ucpMetadata);

  const df = createLargeDataFrame();
  const len = Math.ceil(df.numRows / (workers.length + 1));
  const table = df.toArrow();

  bc.createTable('test_table', DataFrame.fromArrow(table.slice(0, len).serialize()));

  workers.forEach((w, i) => {
    w.send({
      operation: createTable,
      tableName: 'test_table',
      dataframe: table.slice((i + 1) * len, (i + 2) * len).serialize()
    });
  });

  let ctxToken = 0;
  const queryPromises = [];
  const query = 'SELECT a FROM test_table';

  queryPromises.push(new Promise((resolve) => {
    const token = ctxToken++;
    const messageId = `message_${token}`;
    setTimeout(() => {
      const df = bc.sql(query, token).result();
      console.log(`Finished query on token: ${token}`);
      resolve({ ctxToken: token, messageId, df });
    });
  }));

  workers.forEach((w) => {
    queryPromises.push(new Promise((resolve) => {
      const token = ctxToken++;
      const messageId = `message_${token}`;
      w.send({ operation: runQuery, ctxToken: token, messageId, query });
      w.on('message', ({ operation, ctxToken, messageId }) => {
        if (operation === queryRan) {
          console.log(`Finished query on token: ${ctxToken}`);
          console.log(`pulling result with messageId='${messageId}'`);
          resolve({ ctxToken, messageId, df: bc.pullFromCache(messageId) });
        }
      });
    }));
  });

  let result_df = new DataFrame({ a: Series.new([]) });

  Promise.all(queryPromises).then(function (results) {
    console.log('Finished running all queries.');
    results.forEach(({ df, messageId }) => {
      console.log(``);
      console.log(`df for ${messageId}:`);
      console.log(df.toArrow().toArray());
      console.log(``);
      result_df = result_df.concat(df);
    });
    console.log(``);
    console.log('Final result df:');
    console.log(result_df.toArrow().toArray());
    console.log(``);
    workers.forEach((w) => w.kill());
  });

} else if (cluster.isWorker) {
  process.on('message', (args) => {

    const { operation, ...rest } = args;
    const {
      ctxToken, dataframe, messageId, query, tableName, ucpMetadata
    } = rest;

    console.log(`message "${operation}":`, rest);

    if (operation === createBlazingContext) {
      bc = createContext(cluster.worker.id, ucpMetadata);
    }

    if (operation === createTable) {
      bc.createTable(tableName, DataFrame.fromArrow(dataframe));
    }

    if (operation === runQuery) {
      bc.sql(query, ctxToken).sendTo(0, messageId);
      process.send({ operation: queryRan, ctxToken, messageId });
    }
  });
}

function createLargeDataFrame() {
  const a = Series.new(Array.from(Array(300).keys()));
  return new DataFrame({ 'a': a, 'b': a });
}

function createContext(id, ucpMetadata) {
  return new BlazingContext({
    ralId: id,
    enableLogging: true,
    ralCommunicationPort: 4000 + id,
    configOptions: { ...configOptions },
    // networkIfaceName: 'eno1',
    workersUcpInfo: ucpMetadata.map((xs) => ({ ...xs, ucpContext })),
  });
}
