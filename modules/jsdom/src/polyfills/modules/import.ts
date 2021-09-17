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

import * as fs from 'fs';
import * as vm from 'vm';

import {Require} from './require';

const {SourceTextModule, SyntheticModule} = <any>vm;

export function createImport(require: Require,
                             context: import('vm').Context,
                             transform: (path: string, code: string) => string | null | undefined) {
  return importModuleDynamically;

  async function importModuleDynamically(specifier: string) {
    const path = require.resolve(specifier);
    const opts = {displayErrors: true, context, importModuleDynamically, identifier: path};
    try {
      // Try importing via require first
      return await tryRequire(path, opts);
    } catch (e) {
      // If require throws, try importing as ESM
      return await tryImport(path, opts);
    }
  }

  function tryRequire(path: string, opts: any) {
    const exports = require(path);
    if (exports.__esModule && !('default' in exports)) {  //
      exports.default = exports;
    }
    const keys = Object.keys(exports).filter((name) => name !== '__esModule');
    return linkAndEvaluate(new SyntheticModule(keys, function(this: any) {  //
      keys.forEach((n) => this.setExport(n, exports[n]));
    }, opts));
  }

  function tryImport(path: string, opts: any) {
    return linkAndEvaluate(
      new SourceTextModule(transform(path, fs.readFileSync(path, 'utf8')), opts));
  }

  function linkAndEvaluate(module: any) {
    return module.link(importModuleDynamically).then(() => module.evaluate());
  }
}
