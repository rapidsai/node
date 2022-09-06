// Copyright (c) 2022, NVIDIA CORPORATION.
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

const assert        = require('assert');
const {RapidsJSDOM} = require('@rapidsai/jsdom');

(async () => {  //
  const jsdom = new RapidsJSDOM({glfwOptions: {visible: false}});
  await jsdom.loaded;
  assert((await jsdom.window.evalFn(async () => {
           const {importedModuleSharesGlobalsWithThisModule} =
             (await import('./files/test-esm-import')).default;
           return importedModuleSharesGlobalsWithThisModule;
         })) === true,
         'test-esm-import and test-esm-module should share globals');

  assert((await jsdom.window.evalFn(async () => {
           const {importedModuleSharesGlobalsWithThisModule} =
             (await import('./files/test-cjs-import'));
           return importedModuleSharesGlobalsWithThisModule;
         })) === true,
         'test-cjs-import and test-cjs-module should share globals');

  return 0;
})()
  .catch((e) => {
    console.error(e);
    return 1;
  })
  .then(code => process.exit(code ?? 0));
