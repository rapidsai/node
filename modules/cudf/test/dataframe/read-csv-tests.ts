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

import {setDefaultAllocator} from '@nvidia/cuda';
import {DataFrame} from '@nvidia/cudf';
import {DeviceBuffer} from '@nvidia/rmm';

import {mkdtempSync, promises} from 'fs';
import * as Path from 'path';

import {makeCSVString} from './utils';

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength));

describe('DataFrame.readCSV', () => {
  test('can read a CSV string', () => {
    const rows = [
      {a: 0, b: 1.0, c: "2"},
      {a: 1, b: 2.0, c: "3"},
      {a: 2, b: 3.0, c: "4"},
    ];
    const df = DataFrame.readCSV({
      header: 0,
      sourceType: 'buffers',
      sources: [Buffer.from(makeCSVString({rows}))],
      dataTypes: {a: "int32", b: "float64", c: "str"},
    });
    expect(df.get('a').toArrow().values).toEqual(new Int32Array([0, 1, 2]));
    expect(df.get('b').toArrow().toArray()).toEqual(new Float64Array([1.0, 2.0, 3.0]));
    expect([...df.get('c').toArrow()]).toEqual(["2", "3", "4"]);
  });

  test('can read a CSV file', async () => {
    const rows = [
      {a: 0, b: 1.0, c: "2"},
      {a: 1, b: 2.0, c: "3"},
      {a: 2, b: 3.0, c: "4"},
    ];
    const path = Path.join(csvTmpDir, 'simple.csv');
    await promises.writeFile(path, makeCSVString({rows}));
    const df = DataFrame.readCSV({
      header: 0,
      sourceType: 'files',
      sources: [path],
      dataTypes: {a: "int32", b: "float64", c: "str"},
    });
    expect(df.get('a').toArrow().values).toEqual(new Int32Array([0, 1, 2]));
    expect(df.get('b').toArrow().toArray()).toEqual(new Float64Array([1.0, 2.0, 3.0]));
    expect([...df.get('c').toArrow()]).toEqual(["2", "3", "4"]);
    await new Promise<void>((r) => rimraf(path, () => r()));
  });
});

let csvTmpDir = '';

const rimraf = require('rimraf');

beforeAll(() => { csvTmpDir = mkdtempSync(Path.join('/tmp', 'node_cudf')); });

afterAll(() => {
  return new Promise<void>((resolve, reject) => {  //
    rimraf(csvTmpDir, (err?: Error|null) => err ? reject(err) : resolve());
  });
});
