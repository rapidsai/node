// Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

import {readFile} from 'fs/promises';
import * as Path from 'path';
import * as vm from 'vm';
import {SourceTextModule, SyntheticModule} from 'vm';

import {createContextRequire as createRequire, ESMSyntheticModule, Require} from './require';

Object.entries({SyntheticModule, SourceTextModule}).forEach(([name, Class]) => {
  if (!Class) {
    console.error(`${name} not found. ` +
                  `@rapidsai/jsdom requires node is with the --experimental-vm-modules flag.`);
    process.exit(1);
  }
});

export function createDynamicImporter(
  require: Require, outerContext: vm.Context, transform: (path: string, code: string) => string) {
  const moduleLinker = createModuleLinker(require, outerContext, transform);

  return importModuleDynamically;

  function importModuleDynamically(
    specifier: string, _script?: vm.Script, assertions: {[key: string]: any} = {}) {
    return moduleLinker(specifier, {context: outerContext}, {assert: assertions}) as any;
  }
}

export function createModuleLinker(
  require: Require, outerContext: vm.Context, transform: (path: string, code: string) => string) {
  return linkModule;
  function linkModule(specifier: string,
                      {context}: {context: vm.Context} = {context: outerContext},
                      {assert: assertions}: {assert: {[key: string]: any}} = {assert: {}}) {
    const opts = {
      context,
      identifier: require.resolve(specifier),
      importModuleDynamically: createDynamicImporter(require, context, transform),
    } as (vm.SyntheticModuleOptions | vm.SourceTextModuleOptions);
    try {
      // Try importing as CJS first
      return tryRequire(opts, assertions);
    } catch (e1: any) {
      try {
        // If CJS throws, try importing as ESM
        return tryImport(opts, assertions);
      } catch (e2: any) {  //
        throw[e1, e2];
      }
    }
  }

  function tryRequire(opts: vm.SyntheticModuleOptions, assertions: {[key: string]: any}) {
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    const path    = opts.identifier!;
    const exports = require(path);
    if (!exports[ESMSyntheticModule]) {
      const keys = Object.keys(exports);
      if (exports.__esModule && !('default' in exports)) {
        exports.default                  = exports;
        keys[keys.indexOf('__esModule')] = 'default';
      }
      const mod = new SyntheticModule(keys, function() {  //
        keys.forEach((name) => this.setExport(name, exports[name]));
      }, opts);
      return linkAndEvaluate(exports[ESMSyntheticModule] = mod, assertions);
    }
    return Promise.resolve(exports[ESMSyntheticModule]);
  }

  function tryImport(opts: vm.SourceTextModuleOptions, assertions: {[key: string]: any}) {
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    const path = opts.identifier!;
    const dir  = Path.dirname(path);
    const ext  = Path.extname(path);
    return readFile(path, 'utf8')
      .then((code) => {
        code = `var __dirname='${dir}';\n${code}`;
        code = `var __filename='${path}';\n${code}`;
        if (ext in require.extensions) {  //
          code = transform(path, code);
        }
        return new SourceTextModule(code, opts);
      })
      .then((module) => linkAndEvaluate(module, assertions));
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  function linkAndEvaluate(module: vm.Module, _assertions: {[key: string]: any}) {
    let r     = require;
    const dir = Path.dirname(module.identifier);
    if (dir !== require.main.path) {
      r = createRequire({
        dir,
        transform,
        parent: require.main,
        extensions: require.extensions,
        resolve: require.main.__resolve,
        context: module.context || outerContext,
      });

      r.main._moduleCache  = require.cache;
      r.main._resolveCache = require.main._resolveCache;
    }

    return module.link(createModuleLinker(r, r.main._context, transform))
      .then(() => module.evaluate())
      .then(() => module);
  }
}
