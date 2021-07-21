const cluster = require('cluster');

const createTable = 'createTable';
const runQuery = 'runQuery';

const numberOfWorkers = 2;

if (cluster.isMaster) {
  const workers = Array(numberOfWorkers);
  for (let i = 0; i < numberOfWorkers; ++i) {
    workers[i] = cluster.fork();
  }

  workers.forEach((w) => {
    w.send({ operation: createTable, tableName: 'test_table' });
  });

  workers.forEach((w) => {
    w.send({ operation: runQuery, ctxToken: 0 });
  });

} else if (cluster.isWorker) {
  process.on('message', (args) => {
    if (args.operation === createTable) {
      console.log(`Creating table: ${args.tableName}`);
    }

    if (args.operation === runQuery) {
      console.log(`Token: ${args.ctxToken}`);
    }
  });
}
