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

import {SQLContext} from '@rapidsai/blazingsql';
import {DataFrame, Float64, Int64, Series, Utf8String} from '@rapidsai/cudf';
import {promises} from 'fs';
import * as Path from 'path';

test('create, list, describe, and drop CSV table', async () => {
  const rows = [
    {a: 0, b: 0.0, c: 'foo'},
    {a: 1, b: 1.1, c: 'bar'},
    {a: 2, b: 2.2, c: 'foo'},
  ];
  const path = Path.join(csvTmpDir, 'simple.csv');
  await promises.writeFile(path, makeCSVString({rows}));

  const sqlContext = new SQLContext();
  sqlContext.createTable('test_table', [path]);

  expect(sqlContext.listTables()).toEqual(['test_table']);

  const tableDescription = sqlContext.describeTable('test_table');
  expect([...tableDescription.keys()]).toEqual(['a', 'b', 'c']);
  expect([...tableDescription.values()]).toEqual([new Int64, new Float64, new Utf8String]);

  sqlContext.dropTable('test_table');
  expect(sqlContext.listTables().length).toEqual(0);
});

test('query CSV table', async () => {
  const rows = [
    {a: 0, b: 0.0, c: 'foo'},
    {a: 1, b: 1.1, c: 'bar'},
    {a: 2, b: 2.2, c: 'foo'},
  ];
  const path = Path.join(csvTmpDir, 'simple.csv');
  await promises.writeFile(path, makeCSVString({rows}));

  const sqlContext = new SQLContext();
  sqlContext.createTable('test_table', [path]);

  const result = new DataFrame({'c': Series.new(['foo', 'bar', 'foo'])});
  await expect(sqlContext.sql('SELECT c FROM test_table').result()).resolves.toStrictEqual(result);
});

let csvTmpDir = '';

const rimraf = require('rimraf');

function makeCSVString(
  opts: {rows?: any[], delimiter?: string, lineTerminator?: string, header?: boolean} = {}) {
  const {rows = [], delimiter = ',', lineTerminator = '\n', header = true} = opts;
  const names = Object.keys(rows.reduce(
    (keys, row) => Object.keys(row).reduce((keys, key) => ({...keys, [key]: true}), keys), {}));
  return [
    ...[header ? names.join(delimiter) : []],
    ...rows.map((row) =>
                  names.map((name) => row[name] === undefined ? '' : row[name]).join(delimiter))
  ].join(lineTerminator) +
         lineTerminator;
}

beforeAll(async () => {  //
  csvTmpDir = await promises.mkdtemp(Path.join('/tmp', 'node_sql'));
});

afterAll(() => {
  return new Promise<void>((resolve, reject) => {  //
    rimraf(csvTmpDir, (err?: Error|null) => err ? reject(err) : resolve());
  });
});
