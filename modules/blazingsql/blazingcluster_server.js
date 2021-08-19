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
const fastify = require('fastify')({ logger: false });

let bc;

fastify.get('/run_query', async (request, reply) => {
  const { sql } = request.query;
  if (sql) {
    const sqlWithoutQuotations = sql.slice(1, -1);
    const df = await bc.sql(sqlWithoutQuotations);
    return df.names.reduce((result, name) => {
      result += `${name}: ${[...df.get(name)]} \n\n`;
      return result;
    }, '');
  }
  return 'SQL table "test_table" created and ready to be queried. Enter the following route... /run_query?sql="SELECT a FROM test_table"';
});

const start = async () => {
  try {
    bc = await BlazingCluster.init(2);
    await bc.createTable('test_table', createLargeDataFrame());
    await fastify.listen(3000);
    console.log("Server initialized and ready to query...");
  } catch (err) {
    fastify.log.error(err);
    process.exit(1);
  }
}

start();

// TODO: Replace with a proper dataset.
function createLargeDataFrame() {
  const a = Series.new(Array.from({ length: 300 }, (_, i) => i + 1));
  const b = Series.new(Array.from({ length: 300 }, (_, i) => i + 5));
  return new DataFrame({ 'a': a, 'b': b });
}
