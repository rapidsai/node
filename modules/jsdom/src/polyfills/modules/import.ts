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

import * as fs from 'fs/promises';
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

declare module 'vm' {
  interface Module {
    context: vm.Context;
    identifier: string;
    evaluate(): Promise<vm.Module>;
    link(linker: (specifier: string, parent?: vm.Module) => vm.Module |
                                                            Promise<vm.Module>): Promise<vm.Module>;
  }

  interface SyntheticModule extends Module {
    setExport(name: string, value: any): void;
  }
  interface SyntheticModuleConstructor {
    new(exportNames: string[],
        evaluateCallback: (this: vm.SyntheticModule) => void,
        options?: string|{identifier: string, context?: vm.Context}): SyntheticModule;
    readonly name: 'SyntheticModule';
  }
  const SyntheticModule: SyntheticModuleConstructor;

  type SourceTextModule = Module
  interface SourceTextModuleConstructor {
    new(code: string,
        options?: string|{identifier: string, context?: vm.Context}): SourceTextModule;
    readonly name: 'SourceTextModule';
  }
  const SourceTextModule: SourceTextModuleConstructor;
}

export function createImport(require: Require,
                             context: import('vm').Context,
                             transform: (path: string, code: string) => string) {
  return importModuleDynamically;

  async function importModuleDynamically(specifier: string, parent?: vm.Script|vm.Module) {
    const path = require.resolve(specifier);
    const opts = {
      identifier: path,
      displayErrors: true,
      importModuleDynamically,
      context: (<any>parent)?.context || context,
    };
    try {
      // Try importing as CJS first
      return await tryRequire(path, opts);
    } catch (e1: any) {
      try {
        // If CJS throws, try importing as ESM
        return await tryImport(path, opts);
      } catch (e2: any) {  //
        throw[e1, e2];
      }
    }
  }

  function tryRequire(path: string, opts: any) {
    const exports = (require as any)(path, true);
    const keys    = Object.keys(exports);
    if (exports.__esModule && !('default' in exports)) {
      exports.default                  = exports;
      keys[keys.indexOf('__esModule')] = 'default';
    }
    const mod = new SyntheticModule(keys, function() {  //
      keys.forEach((name) => this.setExport(name, exports[name]));
    }, opts);
    return linkAndEvaluate(exports[ESMSyntheticModule] = mod);
  }

  function tryImport(path: string, opts: any) {
    const dir = Path.dirname(path);
    const ext = Path.extname(path);
    return fs.readFile(path, 'utf8')
      .then((code) => {
        code = `var __dirname='${dir}';\n${code}`;
        code = `var __filename='${path}';\n${code}`;
        if (ext in require.extensions) {  //
          code = transform(path, code);
        }
        return new SourceTextModule(code, opts);
      })
      .then(linkAndEvaluate);
  }

  function linkAndEvaluate(module: vm.Module) {
    let r     = require;
    const dir = Path.dirname(module.identifier);
    if (dir !== require.main.path) {
      r = createRequire({
        dir,
        transform,
        parent: require.main,
        extensions: require.extensions,
        resolve: require.main.__resolve,
        context: module.context || context,
      });

      r.main._moduleCache  = require.cache;
      r.main._resolveCache = require.main._resolveCache;
    }

    return module.link(createImport(r, r.main._context, transform))
      .then(() => module.evaluate())
      .then(() => module);
  }
}
