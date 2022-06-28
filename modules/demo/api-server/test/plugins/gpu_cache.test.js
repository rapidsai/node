// Copyright (c) 2022, NVIDIA CORPORATION.
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

'use strict'

const {test}   = require('tap')
const Fastify  = require('fastify')
const Support  = require('../../plugins/support')
const fixtures = require('../fixtures.js');
const gpuCache = require('../../util/gpu_cache.js');

test('clearCachedGPUData()', async t => {
  await gpuCache.setDataframe('bob', 5);
  const result = await gpuCache.getDataframe('bob');
  t.equal(result, 5);
});

test('read_large_graph_demo', async t => {
  const dir    = t.testdir(fixtures);
  const result = await gpuCache.readLargeGraphDemo(dir + '/json_large/json_large.txt');
  await gpuCache.clearDataframes();
  t.same(Object.keys(result), ['nodes', 'edges', 'options']);
});

test('readGraphology', async t => {
  const dir    = t.testdir(fixtures);
  const result = await gpuCache.readGraphology(dir + '/json_good/json_good.txt');
  await gpuCache.clearDataframes();
  t.same(Object.keys(result), ['nodes', 'edges', 'tags', 'clusters']);
});
