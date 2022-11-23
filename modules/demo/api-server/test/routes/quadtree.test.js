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

const {dir}                                   = require('console');
const {test}                                  = require('tap');
const {build}                                 = require('../helper');
const {tableFromIPC, RecordBatchStreamWriter} = require('apache-arrow');
const {csv_quadtree}                          = require('../fixtures.js');
const gpu_cache                               = require('../../util/gpu_cache');

test('quadtree/create/:table', async (t) => {
  const dir   = t.testdir(csv_quadtree);
  const rpath = 'test/routes/' + dir.substring(dir.lastIndexOf('/'));
  const app   = await build(t);
  gpu_cache._setPathForTesting(rpath);
  const load = await app.inject(
    {method: 'POST', url: '/gpu/DataFrame/readCSV', body: {filename: 'csv_quadtree.csv'}});
  const res     = await app.inject({
    method: 'POST',
    url: '/quadtree/create/csv_quadtree.csv',
    body: {xAxisName: 'x', yAxisName: 'y'}
  });
  const release = await app.inject({method: 'POST', url: '/graphology/release'});
  const result  = JSON.parse(res.payload);
  t.same(result, {
    statusCode: 200,
    message: 'Quadtree created',
    params: {table: 'csv_quadtree.csv'},
    success: true
  })
});

test('quadtree/set_polygons', {only: true}, async (t) => {
  const dir   = t.testdir(csv_quadtree);
  const rpath = 'test/routes/' + dir.substring(dir.lastIndexOf('/'));
  const app   = await build(t);
  gpu_cache._setPathForTesting(rpath);
  const load = await app.inject(
    {method: 'POST', url: '/gpu/DataFrame/readCSV', body: {filename: 'csv_quadtree.csv'}});
  const res     = await app.inject({
    method: 'POST',
    url: '/quadtree/set_polygons_quadtree',
    body:
      {name: 'test', polygon_offset: [0, 1], ring_offset: [0, 4], points: [0, 0, 1, 1, 2, 2, 3, 3]}
  });
  const release = await app.inject({method: 'POST', url: '/graphology/release'});
  const result  = JSON.parse(res.payload);
  t.same(result, {
    statusCode: 200,
    message: 'Set polygon test',
    params:
      {name: 'test', polygon_offset: [0, 1], ring_offset: [0, 4], points: [0, 0, 1, 1, 2, 2, 3, 3]},
    success: true
  })
});
