// Copyright (c) 2023, NVIDIA CORPORATION.
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

test('rapids-viewer/set_dataframe', async t => {
  const dir   = t.testdir(csv_quadtree);
  const rpath = 'test/routes/' + dir.substring(dir.lastIndexOf('/'));
  const app   = await build(t);
  gpu_cache._setPathForTesting(rpath);
  const load =
    (await app.inject(
       {method: 'POST', url: '/gpu/DataFrame/readCSV', body: {filename: 'csv_quadtree.csv'}}))
      .json();
  const res     = await app.inject({
    method: 'POST',
    url: '/rapids-viewer/set_dataframe',
    body: {dataframe: load.params.filename, xAxisName: 'Longitude', yAxisName: 'Latitude'}
  });
  const release = await app.inject({method: 'POST', url: '/gpu/release'});
  t.same(res.statusCode, 200);
  t.same(res.json(), {success: true});
});

test('rapids-viewer/set_viewport', async t => {
  const dir   = t.testdir(csv_quadtree);
  const rpath = 'test/routes/' + dir.substring(dir.lastIndexOf('/'));
  const app   = await build(t);
  gpu_cache._setPathForTesting(rpath);
  const load =
    (await app.inject(
       {method: 'POST', url: '/gpu/DataFrame/readCSV', body: {filename: 'csv_quadtree.csv'}}))
      .json();
  const res = await app.inject(
    {method: 'POST', url: '/rapids-viewer/set_viewport', body: {lb: [0, 0], ub: [1, 1]}});
  const release = await app.inject({method: 'POST', url: '/gpu/release'});
  t.same(res.statusCode, 200);
  t.same(res.json(), {success: true});
});

test('rapids-viewer/change_budget', async t => {
  const dir   = t.testdir(csv_quadtree);
  const rpath = 'test/routes/' + dir.substring(dir.lastIndexOf('/'));
  const app   = await build(t);
  gpu_cache._setPathForTesting(rpath);
  const load =
    (await app.inject(
       {method: 'POST', url: '/gpu/DataFrame/readCSV', body: {filename: 'csv_quadtree.csv'}}))
      .json();
  const res =
    await app.inject({method: 'POST', url: '/rapids-viewer/change_budget', body: {budget: 1000}});
  const release = await app.inject({method: 'POST', url: '/gpu/release'});
  t.same(res.statusCode, 200);
  t.same(res.json(), {success: true});
});

test('rapids-viewer/get_n', async t => {
  const dir   = t.testdir(csv_quadtree);
  const rpath = 'test/routes/' + dir.substring(dir.lastIndexOf('/'));
  const app   = await build(t);
  gpu_cache._setPathForTesting(rpath);
  const load =
    (await app.inject(
       {method: 'POST', url: '/gpu/DataFrame/readCSV', body: {filename: 'csv_quadtree.csv'}}))
      .json();
  const vp      = await app.inject({
    method: 'POST',
    url: '/rapids-viewer/set_viewport',
    body: {lb: [-1000, -1000], ub: [1000, 1000]}
  });
  const res     = await app.inject({method: 'GET', url: '/rapids-viewer/get_n/10'});
  const release = await app.inject({method: 'POST', url: '/gpu/release'});
  t.same(res.statusCode, 200);
  const table          = tableFromIPC(res.rawPayload);
  const got            = table.getChild('points_in_polygon').toArray();
  const expectedLength = 18;
  t.same(expectedLength, got.length);
});

test('rapids-viewer/get_n empty the budget', async t => {
  const dir   = t.testdir(csv_quadtree);
  const rpath = 'test/routes/' + dir.substring(dir.lastIndexOf('/'));
  const app   = await build(t);
  gpu_cache._setPathForTesting(rpath);
  const load =
    (await app.inject(
       {method: 'POST', url: '/gpu/DataFrame/readCSV', body: {filename: 'csv_quadtree.csv'}}))
      .json();
  const vp = await app.inject({
    method: 'POST',
    url: '/rapids-viewer/set_viewport',
    body: {lb: [-1000, -1000], ub: [1000, 1000]}
  });
  let res  = await app.inject({method: 'GET', url: '/rapids-viewer/get_n/4'});
  t.same(res.statusCode, 200);
  let table          = tableFromIPC(res.rawPayload);
  let got            = table.getChild('points_in_polygon').toArray();
  let expectedLength = 8;
  t.same(expectedLength, got.length);
  res = await app.inject({method: 'GET', url: '/rapids-viewer/get_n/4'});
  t.same(res.statusCode, 200);
  table          = tableFromIPC(res.rawPayload);
  got            = table.getChild('points_in_polygon').toArray();
  expectedLength = 8;
  t.same(expectedLength, got.length);
  res = await app.inject({method: 'GET', url: '/rapids-viewer/get_n/4'});
  t.same(res.statusCode, 200);
  table          = tableFromIPC(res.rawPayload);
  got            = table.getChild('points_in_polygon').toArray();
  expectedLength = 2;
  t.same(expectedLength, got.length);
  res = await app.inject({method: 'GET', url: '/rapids-viewer/get_n/4'});
  t.same(res.statusCode, 200);
  table          = tableFromIPC(res.rawPayload);
  got            = table.getChild('points_in_polygon').toArray();
  expectedLength = 0;
  t.same(expectedLength, got.length);
  const release = await app.inject({method: 'POST', url: '/gpu/release'});
});
