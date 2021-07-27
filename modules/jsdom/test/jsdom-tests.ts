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

import {RapidsJSDOM} from '@rapidsai/jsdom';
import {evalAsync} from './utils';

test('fails to require a non-existent file', async () => {
  const {window} = new RapidsJSDOM();
  await expect(evalAsync(window, () => {  //
    return typeof require(`./files/nonexistent_file`) === 'object';
  })).rejects.toThrow();
});

test('successfully requires a local CommonJS module', async () => {
  const {window} = new RapidsJSDOM();
  await expect(evalAsync(window, () => {  //
    return typeof require(`./files/test-cjs-module`) === 'object';
  })).resolves.toBe(true);
});

test('successfully requires a local ESModule module', async () => {
  const {window} = new RapidsJSDOM();
  const result   = evalAsync(window, () => {  //
    return typeof require(`./files/test-esm-module`).default === 'object';
  });
  await expect(result).resolves.toBe(true);
});
