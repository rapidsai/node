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

const { BlazingCluster } = require('@rapidsai/blazingsql');
const { Series, DataFrame } = require('@rapidsai/cudf');
const fastify = require('fastify')({});

let bc;
const init = async () => {
  bc = await BlazingCluster.init(2);
  await bc.createTable('test_table', createLargeDataFrame());
}
init();

// Change cwd to the example dir so relative file paths are resolved
process.chdir(__dirname);

fastify
  .register(require('fastify-nextjs'))
  .after(() => {
    fastify.next('/')
    fastify.get('/run_query', function (request, reply) {
      reply.send({ hello: 'world' })
    })
  });

fastify.listen(3000, err => {
  if (err) throw err
  console.log('Server listening on http://localhost:3000')
});

function createLargeDataFrame() {
  const a = Series.new(Array.from({ length: 300 }, (_, i) => i + 1));
  const b = Series.new(Array.from({ length: 300 }, (_, i) => i + 5));
  return new DataFrame({ 'a': a, 'b': b });
}
