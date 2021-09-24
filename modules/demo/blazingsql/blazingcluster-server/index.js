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
const {BlazingCluster}          = require('@rapidsai/blazingsql');
const {DataFrame}               = require('@rapidsai/cudf');
const {RecordBatchStreamWriter} = require('apache-arrow');
const fs                        = require('fs');

const fastify = require('fastify')({
  pluginTimeout: 30000,
  logger: process.env.NODE_ENV !== 'production',
});

const DATA_PATH = `${__dirname}/wikipedia_pages.csv`;
if (!fs.existsSync(DATA_PATH)) {
  console.error(`
  .csv data not found! Run this to download the dataset from AWS S3 (16.0 GB):

  node ${__dirname}/data.js
  `);
  process.exit(1);
}

let bc;

// Change cwd to the example dir so relative file paths are resolved
process.chdir(__dirname);

fastify.register((require('fastify-arrow')))
  .register(require('fastify-nextjs', {
    dev: process.env.NODE_ENV !== 'production',
  }))
  .register(async (instane, opts, done) => {
    bc = await BlazingCluster.init({numWorkers: 2});
    await bc.createTable('test_table', DataFrame.readCSV({
      header: 0,
      sourceType: 'files',
      sources: [DATA_PATH],
    }));
    done();
  })
  .after(() => {
    fastify.next('/');
    fastify.post('/run_query', async function(request, reply) {
      const t0         = performance.now();
      const df         = await bc.sql(request.body);
      const t1         = performance.now();
      const queryTime  = t1 - t0;
      const arrowTable = df.toArrow();
      arrowTable.schema.metadata.set('queryTime', queryTime);
      RecordBatchStreamWriter.writeAll(arrowTable).pipe(reply.stream());
    });
  });

fastify.listen(3000, err => {
  if (err) throw err
    console.log('Server listening on http://localhost:3000')
});
