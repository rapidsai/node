// Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

import {globalWindow} from './utils';

const eval_ = (code: string) => globalWindow.evalFn(() => {
  const res = eval(code);
  return res;
}, {code});

test('fails to require a non-existent file', () => {
  expect(() => globalWindow.eval(`require('./files/nonexistent_file')`))  //
    .toThrow();
});

test('requires a local CommonJS module', () => {
  const code = `require('./files/test-cjs-module')`;
  expect(typeof eval_(code)).toBe('object');
});

// this requires we transpile ESM to CJS with babel
test.skip('requires a local ESModule module', () => {
  const code = `require('./files/test-esm-module').default`;
  expect(typeof eval_(code)).toBe('object');
});

test('imports a local CommonJS module', async () => {
  const code = `import('./files/test-cjs-module')`;
  expect(typeof (await eval_(code))).toBe('object');
});

test('imports a local ESModule module', async () => {
  const code = `import('./files/test-esm-module')`;
  expect(typeof (await eval_(code)).default).toBe('object');
});

test('requires a local CommonJS module that imports an ESModule', () => {
  const code                                        = `require('./files/test-cjs-import')`;
  const {importedModuleSharesGlobalsWithThisModule} = eval_(code);
  expect(importedModuleSharesGlobalsWithThisModule).toBe(true);
});

// this requires we transpile ESM to CJS with babel
test.skip('requires a local ESModule module that imports an ESModule', () => {
  const code                                        = `require('./files/test-esm-import')`;
  const {importedModuleSharesGlobalsWithThisModule} = eval_(code).default;
  expect(importedModuleSharesGlobalsWithThisModule).toBe(true);
});

test('imports a local CommonJS module that imports an ESModule', async () => {
  const code                                        = `import('./files/test-cjs-import')`;
  const {importedModuleSharesGlobalsWithThisModule} = await eval_(code);
  expect(importedModuleSharesGlobalsWithThisModule).toBe(true);
});

test('imports a local ESModule module that imports an ESModule', async () => {
  const code                                        = `import('./files/test-esm-import')`;
  const {importedModuleSharesGlobalsWithThisModule} = (await eval_(code)).default;
  expect(importedModuleSharesGlobalsWithThisModule).toBe(true);
});

test('CJS modules imported as ESM modules can modify their named exports ', async () => {
  const code          = `import('./files/test-cjs-module')`;
  const testCJSModule = (await eval_(code)) as typeof import('./files/test-cjs-module');
  expect(typeof testCJSModule).toBe('object');
  expect(testCJSModule.foo).toEqual('foo');
  testCJSModule.setFooToBar();
  expect(testCJSModule.foo).toEqual('bar');
});
