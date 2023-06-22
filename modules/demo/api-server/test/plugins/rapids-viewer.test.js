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

const {test}                     = require('tap')
const Fastify                    = require('fastify')
const Support                    = require('../../plugins/support')
const fixtures                   = require('../fixtures.js');
const RapidsViewer               = require('../../util/rapids-viewer.js');
const {DataFrame, Series, Int32} = require('@rapidsai/cudf');

test('set_df', async t => {
  const df        = new DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [6, 7, 8, 9, 10],
  });
  const xAxisName = 'x';
  const yAxisName = 'y';
  RapidsViewer.set_df(df, xAxisName, yAxisName);
  t.same(RapidsViewer.df, df);
  t.same(RapidsViewer.xAxisName, xAxisName);
  t.same(RapidsViewer.yAxisName, yAxisName);
});

test('set_viewport', async t => {
  const df        = new DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [6, 7, 8, 9, 10],
  });
  const xAxisName = 'x';
  const yAxisName = 'y';
  RapidsViewer.set_df(df, xAxisName, yAxisName);
  const lb = [-1, -1];
  const ub = [11, 11];
  RapidsViewer.set_viewport(lb, ub);
  t.same(RapidsViewer.lb, lb);
  t.same(RapidsViewer.ub, ub);
  t.ok(RapidsViewer.quadtree);
});

test('change_budget', async t => {
  const budget = 5000;
  RapidsViewer.change_budget(5000);
  t.same(RapidsViewer.budget, budget);
});

test('get_n', async t => {
  const result = RapidsViewer.get_n(5);
  t.same(result.numRows, 10);
});

test('empty the budget', async t => {
  RapidsViewer.set_df(new DataFrame({
                        'x': Series.sequence({type: new Int32, size: 100}),
                        'y': Series.sequence({type: new Int32, size: 100}),
                      }),
                      'x',
                      'y');
  RapidsViewer.set_viewport([0, 0], [100, 100]);
  RapidsViewer.change_budget(100);
  let current = RapidsViewer.get_n(10);
  let gotten  = current.numRows;
  while (current.numRows > 0) {
    current = RapidsViewer.get_n(10);
    gotten += current.numRows;
  }
  t.same(gotten, 200);
});

test('empty the budget of half of the available points', async t => {
  RapidsViewer.set_df(new DataFrame({
                        'x': Series.sequence({type: new Int32, size: 100}),
                        'y': Series.sequence({type: new Int32, size: 100}),
                      }),
                      'x',
                      'y');
  RapidsViewer.set_viewport([0, 0], [100, 100]);
  RapidsViewer.change_budget(50);
  let current = RapidsViewer.get_n(10);
  let gotten  = current.numRows;
  while (current.numRows > 0) {
    current = RapidsViewer.get_n(10);
    gotten += current.numRows;
  }
  t.same(gotten, 100);
});

test('get points from a viewport until they run out', {only: true}, async t => {
  RapidsViewer.set_df(new DataFrame({
                        'x': Series.sequence({type: new Int32, size: 100}),
                        'y': Series.sequence({type: new Int32, size: 100}),
                      }),
                      'x',
                      'y');
  RapidsViewer.set_viewport([0, 0], [30, 30]);
  RapidsViewer.change_budget(50);
  let current = RapidsViewer.get_n(10);
  let gotten  = current.numRows;
  while (current.numRows > 0) {
    current = RapidsViewer.get_n(10);
    gotten += current.numRows;
  }
  t.same(gotten, 60);
});

test('change the viewport and get budget points again', async t => {
  RapidsViewer.set_df(new DataFrame({
                        'x': Series.sequence({type: new Int32, size: 100}),
                        'y': Series.sequence({type: new Int32, size: 100}),
                      }),
                      'x',
                      'y');
  RapidsViewer.set_viewport([0, 0], [30, 30]);
  RapidsViewer.change_budget(50);
  let current = RapidsViewer.get_n(10);
  let gotten  = current.numRows;
  while (current.numRows > 0) {
    current = RapidsViewer.get_n(10);
    gotten += current.numRows;
  }
  t.same(gotten, 60);
  RapidsViewer.set_viewport([0, 0], [100, 100]);
  current = RapidsViewer.get_n(10);
  gotten  = current.numRows;
  while (current.numRows > 0) {
    current = RapidsViewer.get_n(10);
    gotten += current.numRows;
  }
  t.same(gotten, 100);
});

test('increase the budget without changing the viewport', async t => {
  RapidsViewer.set_df(new DataFrame({
                        'x': Series.sequence({type: new Int32, size: 100}),
                        'y': Series.sequence({type: new Int32, size: 100}),
                      }),
                      'x',
                      'y');
  RapidsViewer.change_budget(50);
  let current = RapidsViewer.get_n(10);
  let gotten  = current.numRows;
  while (current.numRows > 0) {
    current = RapidsViewer.get_n(10);
    gotten += current.numRows;
  }
  t.same(gotten, 100);
  RapidsViewer.change_budget(100);
  current = RapidsViewer.get_n(10);
  gotten  = current.numRows;
  while (current.numRows > 0) {
    current = RapidsViewer.get_n(10);
    gotten += current.numRows;
  }
  t.same(gotten, 200);
});
