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
  const result = await bc.sql('SELECT a FROM test_table');
  const result2 = await bc.sql('SELECT b FROM test_table');

  bc.stop();

  console.log([...result.get('a')]);
  console.log([...result2.get('b')]);
}

main();

function createLargeDataFrame() {
  const a = Series.new(Array.from(Array(300).keys()));
  return new DataFrame({ 'a': a, 'b': a });
}
