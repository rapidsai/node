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

import {DataFrame, Float64, Series, Utf8String} from '@rapidsai/cudf';
import {SQLCluster} from '@rapidsai/sql';

let sqlCluster: SQLCluster;

beforeAll(async () => { sqlCluster = await SQLCluster.init({numWorkers: 2}); });

afterAll(() => { sqlCluster?.kill(); });

test('create and drop table', async () => {
  const a  = Series.new([1, 2, 3]);
  const df = new DataFrame({'a': a});

  await sqlCluster.createTable('test_table', df);
  expect(sqlCluster.listTables().length).toEqual(1);

  await sqlCluster.dropTable('test_table');
  expect(sqlCluster.listTables().length).toEqual(0);
});

test('list tables', async () => {
  const a  = Series.new([1, 2, 3]);
  const df = new DataFrame({'a': a});

  await sqlCluster.createTable('test_table', df);
  await sqlCluster.createTable('test_table2', df);

  expect(sqlCluster.listTables()).toEqual(['test_table', 'test_table2']);
});

test('describe table', async () => {
  const a  = Series.new([1, 2, 3]);
  const b  = Series.new(['foo', 'bar', 'foo']);
  const df = new DataFrame({'a': a, 'b': b});

  // Empty map since table doesn't exist
  expect(sqlCluster.describeTable('nonexisting_table').size).toEqual(0);

  await sqlCluster.createTable('test_table', df);
  const tableDescription = sqlCluster.describeTable('test_table');
  expect([...tableDescription.keys()]).toEqual(['a', 'b']);
  expect([...tableDescription.values()]).toEqual([new Float64, new Utf8String]);
});

test('explain', async () => {
  const key = Series.new(['a', 'b', 'c', 'd', 'e']);
  const val = Series.new([7.6, 2.9, 7.1, 1.6, 2.2]);
  const df  = new DataFrame({'key': key, 'val': val});

  await sqlCluster.createTable('test_table', df);

  const query = 'SELECT * FROM test_table WHERE val > 4';

  expect(sqlCluster.explain(query))
    .toEqual(
      `LogicalProject(key=[$0], val=[$1])
  BindableTableScan(table=[[main, test_table]], filters=[[>($1, 4)]])
`);
  expect(sqlCluster.explain(query, true))
    .toEqual(
      `LogicalProject(key=[$0], val=[$1])
  BindableTableScan(table=[[main, test_table]], filters=[[>($1, 4)]])

`);
});

test('select a single column (one worker)', async () => {
  const a  = Series.new([6, 9, 1, 6, 2]);
  const b  = Series.new([7, 2, 7, 1, 2]);
  const df = new DataFrame({'a': a, 'b': b});

  await sqlCluster.createTable('test_table', df);

  await expect(sqlCluster.sql('SELECT a FROM test_table'))
    .resolves.toStrictEqual(new DataFrame({a}));
});

test('select all columns (one worker)', async () => {
  const a  = Series.new([6, 9, 1, 6, 2]);
  const b  = Series.new([7, 2, 7, 1, 2]);
  const df = new DataFrame({'a': a, 'b': b});

  await sqlCluster.createTable('test_table', df);

  await expect(sqlCluster.sql('SELECT * FROM test_table'))
    .resolves.toStrictEqual(new DataFrame({'a': a, 'b': b}));
});

test('union columns from two tables (one worker)', async () => {
  const a   = Series.new([1, 2, 3]);
  const df1 = new DataFrame({'a': a});
  const df2 = new DataFrame({'a': a});

  await sqlCluster.createTable('t1', df1);
  await sqlCluster.createTable('t2', df2);

  const result = new DataFrame({'a': Series.new([...a, ...a])});
  await expect(sqlCluster.sql('SELECT a FROM t1 AS a UNION ALL SELECT a FROM t2'))
    .resolves.toStrictEqual(result);
});

test('find all columns within a table that meet condition (one worker)', async () => {
  const key = Series.new(['a', 'b', 'c', 'd', 'e']);
  const val = Series.new([7.6, 2.9, 7.1, 1.6, 2.2]);
  const df  = new DataFrame({'key': key, 'val': val});

  await sqlCluster.createTable('test_table', df);

  const result = new DataFrame({'key': Series.new(['a', 'b']), 'val': Series.new([7.6, 7.1])});
  await expect(sqlCluster.sql('SELECT * FROM test_table WHERE val > 4'))
    .resolves.toStrictEqual(result);
});

test('empty sql result', async () => {
  const key = Series.new(['a', 'b', 'c', 'd', 'e']);
  const val = Series.new([7.6, 2.9, 7.1, 1.6, 2.2]);
  const df  = new DataFrame({'key': key, 'val': val});

  await sqlCluster.createTable('test_table', df);

  const result = new DataFrame();
  // Query should be empty since BETWEEN values are reversed.
  await expect(sqlCluster.sql('SELECT * FROM test_table WHERE val BETWEEN 10 AND 0'))
    .resolves.toStrictEqual(result);
});
