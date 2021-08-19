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

async function main() {
  const df = createLargeDataFrame();

  const bc = await BlazingCluster.init(2);
  await bc.createTable('test_table', df);
  const result = await bc.sql('SELECT a FROM test_table');;

  bc.stop();

  result.names.forEach((n) => {
    console.log(`${n}: ${[...result.get(n)]}`);
  });
}

main();

function createLargeDataFrame() {
  const a = Series.new(Array.from({ length: 300 }, (_, i) => i + 1));
  const b = Series.new(Array.from({ length: 300 }, (_, i) => i + 5));
  return new DataFrame({ 'a': a, 'b': b });
}
