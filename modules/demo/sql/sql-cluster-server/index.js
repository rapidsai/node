#!/usr/bin/env node

// Copyright (c) 2021-2022, NVIDIA CORPORATION.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

const fs                        = require('fs');
const {performance}             = require('perf_hooks');
const {RecordBatchStreamWriter} = require('apache-arrow');
const {SQLCluster}              = require('@rapidsai/sql');
const {DataFrame, scope}        = require('@rapidsai/cudf');

const fastify = require('fastify')({
  pluginTimeout: 30000,
  logger: process.env.NODE_ENV !== 'production',
  keepAliveTimeout: 0,
});

const DATA_PATHS = Array.from({length: 10}, (_, i) => `${__dirname}/data/wiki_page_${i}.csv`);
DATA_PATHS.forEach((DATA_PATH) => {
  if (!fs.existsSync(DATA_PATH)) {
    console.error(`
    .csv data not found! Run this to download the dataset from AWS S3 (16.0 GB):

    node ${__dirname}/data.js
    `);
    process.exit(1);
  }
})

let sqlCluster;

// Change cwd to the example dir so relative file paths are resolved
process.chdir(__dirname);

fastify.register((require('fastify-arrow')))
  .register(require('fastify-nextjs', {
    dev: process.env.NODE_ENV !== 'production',
  }))
  .register(async (instance, opts, done) => {
    const logPath = `${__dirname}/.logs`;
    require('rimraf').sync(logPath);
    fs.mkdirSync(logPath, {recursive: true});

    sqlCluster = await SQLCluster.init({
      numWorkers: Infinity,
      enableLogging: true,
      // allocationMode: 'pool_memory_resource',
      configOptions: {
        PROTOCOL: 'TCP',
        ENABLE_TASK_LOGS: true,
        ENABLE_COMMS_LOGS: true,
        ENABLE_OTHER_ENGINE_LOGS: true,
        ENABLE_GENERAL_ENGINE_LOGS: true,
        BLAZING_LOGGING_DIRECTORY: logPath,
        BLAZING_LOCAL_LOGGING_DIRECTORY: logPath,
      }
    });
    await sqlCluster.createCSVTable('test_table', DATA_PATHS);
    done();
  })
  .after(() => {
    fastify.next('/');
    fastify.post('/run_query', async function(request, reply) {
      try {
        request.log.info({query: request.body}, `calling sqlCluster.sql()`);
        await scope(async () => {
          const t0        = performance.now();
          const dfs       = await toArray(sqlCluster.sql(request.body)).catch((err) => {
            request.log.error({err}, `Error calling sqlCluster.sql`);
            return new DataFrame();
          });
          const t1        = performance.now();
          const queryTime = t1 - t0;

          const {result, rowCount} = head(dfs, 500);
          const arrowTable         = result.toArrow();
          arrowTable.schema.metadata.set('queryTime', queryTime);
          arrowTable.schema.metadata.set('queryResults', rowCount);
          RecordBatchStreamWriter.writeAll(arrowTable).pipe(reply.stream());
        });
      } catch (err) {
        request.log.error({err}, '/run_query error');
        reply.code(500).send(err);
      }
    });
  });

fastify.listen(3000, '0.0.0.0', err => {
  if (err) throw err
    console.log('Server listening on http://localhost:3000')
});

function head(dfs, rows) {
  let result   = new DataFrame();
  let rowCount = 0;

  for (let i = 0; i < dfs.length; ++i) {
    if (dfs[i].numRows > 0) {
      rowCount += dfs[i].numRows;
      if (result.numRows <= rows) {
        result = scope(() => {  //
          return result.concat(dfs[i].head(rows - result.numRows));
        });
      }
    }
  }

  return {result, rowCount};
}

async function toArray(src) {
  const dfs = [];
  for await (const df of src) { dfs.push(df); }
  return dfs;
}
