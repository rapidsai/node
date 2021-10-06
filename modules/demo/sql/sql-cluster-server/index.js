#!/usr/bin/env -S node -r esm

// Copyright (c) 2021, NVIDIA CORPORATION.
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

const {performance}             = require('perf_hooks');
const {SQLCluster}              = require('@rapidsai/sql');
const {DataFrame}               = require('@rapidsai/cudf');
const {RecordBatchStreamWriter} = require('apache-arrow');
const fs                        = require('fs');

const fastify = require('fastify')({
  pluginTimeout: 30000,
  logger: process.env.NODE_ENV !== 'production',
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
    sqlCluster = await SQLCluster.init({numWorkers: 10});
    await sqlCluster.createTable('test_table', DATA_PATHS);
    done();
  })
  .after(() => {
    fastify.next('/');
    fastify.post('/run_query', async function(request, reply) {
      const t0           = performance.now();
      const df           = await sqlCluster.sql(request.body);
      const t1           = performance.now();
      const queryTime    = t1 - t0;
      const queryResults = df.numRows;
      const arrowTable   = df.head(500).toArrow();
      arrowTable.schema.metadata.set('queryTime', queryTime);
      arrowTable.schema.metadata.set('queryResults', queryResults);
      RecordBatchStreamWriter.writeAll(arrowTable).pipe(reply.stream());
    });
  });

fastify.listen(3000, err => {
  if (err) throw err
    console.log('Server listening on http://localhost:3000')
});
