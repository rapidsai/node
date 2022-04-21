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

import {setDefaultAllocator} from '@rapidsai/cuda';
import {DataFrame, Float64, Int32, Utf8String} from '@rapidsai/cudf';
import {DeviceBuffer} from '@rapidsai/rmm';
import {promises} from 'fs';
import * as Path from 'path';

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength));

describe('DataFrame.readJSON', () => {
  test('can read a JSON string', () => {
    const rows = [
      {a: 0, b: 1.0, c: '2'},
      {a: 1, b: 2.0, c: '3'},
      {a: 2, b: 3.0, c: '4'},
    ];
    const rows_string =
      JSON.stringify(rows).replace('[', '').replace(']', '').replaceAll('},', '}\n');
    console.log(rows_string);
    const df = DataFrame.readJSON({
      sourceType: 'buffers',
      sources: [Buffer.from(rows_string)],
      dataTypes: {a: new Int32, b: new Float64, c: new Utf8String},
    });
    expect(df.get('a').data.toArray()).toEqual(new Int32Array([0, 1, 2]));
    expect(df.get('b').data.toArray()).toEqual(new Float64Array([1.0, 2.0, 3.0]));
    expect([...df.get('c')]).toEqual(['2', '3', '4']);
  });

  test('can read a JSON file', async () => {
    const rows = [
      {a: 0, b: 1.0, c: '2'},
      {a: 1, b: 2.0, c: '3'},
      {a: 2, b: 3.0, c: '4'},
    ];
    const rows_string =
      JSON.stringify(rows).replace('[', '').replace(']', '').replaceAll('},', '}\n');
    console.log(rows_string);
    const path = Path.join(csvTmpDir, 'simple.csv');
    await promises.writeFile(path, rows_string);
    const df = DataFrame.readJSON({
      sourceType: 'files',
      sources: [path],
      dataTypes: {a: new Int32, b: new Float64, c: new Utf8String},
    });
    expect(df.get('a').data.toArray()).toEqual(new Int32Array([0, 1, 2]));
    expect(df.get('b').data.toArray()).toEqual(new Float64Array([1.0, 2.0, 3.0]));
    expect([...df.get('c')]).toEqual(['2', '3', '4']);
    await new Promise<void>((resolve, reject) =>
                              rimraf(path, (err?: Error|null) => err ? reject(err) : resolve()));
  });
});

let csvTmpDir = '';

const rimraf = require('rimraf');

beforeAll(async () => {  //
  csvTmpDir = await promises.mkdtemp(Path.join('/tmp', 'node_cudf'));
});

afterAll(() => {
  return new Promise<void>((resolve, reject) => {  //
    rimraf(csvTmpDir, (err?: Error|null) => err ? reject(err) : resolve());
  });
});
