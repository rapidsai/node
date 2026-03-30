// Copyright (c) 2026, NVIDIA CORPORATION.
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
import {Series} from '@rapidsai/cudf';
import {DeviceBuffer} from '@rapidsai/rmm';
import {promises} from 'fs';
import * as Path from 'path';

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength));

describe('Series.readText', () => {
  test('can read a json file', async () => {
    const rows = [
      {a: 0, b: 1.0, c: '2'},
      {a: 1, b: 2.0, c: '3'},
      {a: 2, b: 3.0, c: '4'},
    ];
    const outputString = JSON.stringify(rows);
    const path         = Path.join(readTextTmpDir, 'simple.txt');
    await promises.writeFile(path, outputString);
    const text = Series.readText(path, '');
    expect(text.toArray()).toEqual(outputString.split(''));
    await new Promise<void>((resolve, reject) =>
                              rimraf(path, (err?: Error|null) => err ? reject(err) : resolve()));
  });
  test('can read a random file', async () => {
    const outputString = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()';
    const path         = Path.join(readTextTmpDir, 'simple.txt');
    await promises.writeFile(path, outputString);
    const text = Series.readText(path, '');
    expect(text.toArray()).toEqual(outputString.split(''));
    await new Promise<void>((resolve, reject) =>
                              rimraf(path, (err?: Error|null) => err ? reject(err) : resolve()));
  });
  test('can read an empty file', async () => {
    const outputString = '';
    const path         = Path.join(readTextTmpDir, 'simple.txt');
    await promises.writeFile(path, outputString);
    const text = Series.readText(path, '');
    expect(text.toArray()).toEqual(outputString.split(''));
    await new Promise<void>((resolve, reject) =>
                              rimraf(path, (err?: Error|null) => err ? reject(err) : resolve()));
  });
});

let readTextTmpDir = '';

const rimraf = require('rimraf');

beforeAll(async () => {  //
  readTextTmpDir = await promises.mkdtemp(Path.join('/tmp', 'node_cudf'));
});

afterAll(() => {
  return new Promise<void>((resolve, reject) => {  //
    rimraf(readTextTmpDir, (err?: Error|null) => err ? reject(err) : resolve());
  });
});
