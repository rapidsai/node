const cluster = require('cluster');
const { BlazingContext, UcpContext } = require('@rapidsai/blazingsql')

const createTable = 'createTable';
const runQuery = 'runQuery';
const queryRan = 'queryRan';

const numberOfWorkers = 2;

if (cluster.isMaster) {
  const workers = Array(numberOfWorkers);
  for (let i = 0; i < numberOfWorkers; ++i) {
    workers[i] = cluster.fork();
  }

  const ucp_context = new UcpContext();
  const bc = new BlazingContext(Object.keys(cluster.workers).map((id) => {
    return {
      workerId: id,
      ip: 'localhost',
      port: 8000,
      ucpContext: ucp_context,
    };
  }));

  workers.forEach((w) => {
    w.send({ operation: createTable, tableName: 'test_table' });
  });

  let ctxToken = 0;
  let queryPromises = [];
  workers.forEach((w) => {
    queryPromises.push(new Promise(function (resolve) {
      w.send({ operation: runQuery, ctxToken: ctxToken++ });
      w.on('message', (args) => {
        console.log(`Finished query on token: ${args.ctxToken}`);
        resolve(args);
      });
    }));
  });

  Promise.all(queryPromises).then(function (result) {
    console.log('Finished running all queries.');
    workers.forEach((w) => w.kill());
  });

} else if (cluster.isWorker) {
  process.on('message', (args) => {
    if (args.operation === createTable) {
      console.log(`Creating table: ${args.tableName}`);
    }

    if (args.operation === runQuery) {
      console.log(`Token: ${args.ctxToken}`);
      process.send({ operation: queryRan, ctxToken: args.ctxToken });
    }
  });
}
