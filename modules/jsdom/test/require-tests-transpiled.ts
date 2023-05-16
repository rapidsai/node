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

import {eval_, transpileESMToCJS} from './utils';

transpileESMToCJS();

// this requires we transpile ESM to CJS with babel
test('requires a local ESModule module', () => {
  const code = `require('./files/test-esm-module').default`;
  expect(typeof eval_(code)).toBe('object');
});

// this requires we transpile ESM to CJS with babel
test('requires a local ESModule module that imports an ESModule', () => {
  const code                                        = `require('./files/test-esm-import')`;
  const {importedModuleSharesGlobalsWithThisModule} = eval_(code).default;
  expect(importedModuleSharesGlobalsWithThisModule).toBe(true);
});
