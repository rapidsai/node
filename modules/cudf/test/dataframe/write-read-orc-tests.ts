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

import {Float64Buffer, Int32Buffer} from '@rapidsai/cuda';
import {DataFrame, Float64, Int32, Series} from '@rapidsai/cudf';
import {promises} from 'fs';
import * as Path from 'path';

test('writes and reads an ORC', () => {
  const expected = new DataFrame({
    a: Series.new({length: 3, type: new Int32, data: new Int32Buffer([1, 2, 3])}),
    b: Series.new({length: 3, type: new Float64, data: new Float64Buffer([1.0, 2.0, 3.0])}),
    c: Series.new(['2', '3', '4']),
  });

  const path = Path.join(tmpDir, 'simple.orc');
  expected.toORC(path);

  const result = DataFrame.readORC({
    sourceType: 'files',
    sources: [path],
  });

  expect(result.toString()).toEqual(expected.toString());
});

let tmpDir = '';

const rimraf = require('rimraf');

beforeAll(async () => {  //
  tmpDir = await promises.mkdtemp(Path.join('/tmp', 'node_cudf'));
});

afterAll(() => {
  return new Promise<void>((resolve, reject) => {  //
    rimraf(tmpDir, (err?: Error|null) => err ? reject(err) : resolve());
  });
});
