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

// import * as fs from 'fs';
import * as jsdom from 'jsdom';
// import * as Module from 'module';
import * as vm from 'vm';

import {Require} from './modules/require';

// const {SourceTextModule, SyntheticModule} = <any>vm;

// const {
//   implSymbol,
//   implForWrapper,
//   wrapperForImpl,
// } = require('jsdom/lib/jsdom/living/generated/utils');

declare module 'jsdom' {
  interface DOMWindow {
    evalFn: (script: () => any, globals?: Record<string, any>) => any;
  }
}

// const nodeGlobals = Object.keys(Object.getOwnPropertyDescriptors(vm.runInNewContext('this')));

// const clone = (obj: any) => Object.create(Object.getPrototypeOf(obj),  //
//                                           Object.getOwnPropertyDescriptors(obj));

export function installJSDOMUtils({require, createContext}: {
  require: Require,
  createContext: (globals?: Record<string, any>) => vm.Context,
}) {
  return (window: jsdom.DOMWindow) => {
    // window.jsdom || (window.jsdom = {});
    // window.jsdom.utils || (window.jsdom.utils = {});
    // window.jsdom.utils.implSymbol     = implSymbol;
    // window.jsdom.utils.implForWrapper = implForWrapper;
    // window.jsdom.utils.wrapperForImpl = wrapperForImpl;
    // window.jsdom.global               = window.jsdom.utils.implForWrapper(window)._globalObject;
    // window.jsdom.global.__cwd         = cwd;

    // const context = createContext(window);
    // const require = createContextRequire({context, dir: cwd, resolve});

    window.jsdom.global.require = require;

    if (window.jsdom.utils.implForWrapper(window.document)._origin === 'null') {
      window.jsdom.utils.implForWrapper(window.document)._origin = '';
    }

    window.evalFn = (f: () => any, globals: Record<string, any> = {}) => {
      const source   = `(${f.toString()}).call(this)`;
      const filename = `evalmachine.<${f.name || 'anonymous'}>`;
      return require.main.exec(source, filename, createContext(globals));
      // const outer    = require.main._context;
      // const inner    = createContext(globals);
      // const options  = {
      //   displayErrors: true,
      //   importModuleDynamically: require.main.createImport(inner)
      // };
      // const source = `(${f.toString()}).call(this)`;
      // const script = new vm.Script(source, options);

      // let result;

      // try {
      //   require.main._context = inner;
      //   result                = script.runInContext(inner);
      // } catch (e) {
      //   require.main._context = outer;
      //   throw e;
      // }
      // require.main._context = outer;

      // return result;

      // async function importModuleDynamically(specifier: string) {
      //   let module: any;
      //   const path = require.resolve(specifier);
      //   const opts = {...options, identifier: path};
      //   try {
      //     // First try importing as CJS
      //     const exports = require(path);
      //     if (exports.__esModule && !('default' in exports)) {  //
      //       exports.default = exports;
      //     }
      //     const keys = Object.keys(exports).filter((name) => name !== '__esModule');
      //     module     = new SyntheticModule(
      //       keys, () => keys.forEach((n) => module.setExport(n, exports[n])), opts);
      //     await module.link(importModuleDynamically).then(() => module.evaluate());
      //   } catch (e) {
      //     // If require throws, try importing via ESM
      //     module = new SourceTextModule(fs.readFileSync(path, 'utf8'), opts);
      //     await module.link(importModuleDynamically).then(() => module.evaluate());
      //   }
      //   return module;
      // }
    };

    return window;
  };

  // function createContext(window: jsdom.DOMWindow, globals: Record<string, any> = {}) {
  //   const context = vm.createContext(Object.assign(  //
  //     Object.create(                                 //
  //       window.jsdom.global,                         //
  //       {
  //         ...Object.getOwnPropertyDescriptors(global),
  //         window: {get() { return window; }, configurable: true, enumerable: true},
  //         global: {get() { return context; }, configurable: true, enumerable: true},
  //       }),
  //     {
  //       process: Object.assign(clone(process), {
  //         __cwd: cwd,
  //         browser: true,
  //         cwd() { return this.__cwd; },
  //         chdir(dir: string) { this.__cwd = dir; },
  //       }),
  //       ...globals,
  //     }));

  //   return installSymbolHasInstanceImpls(context);
  // }
}

// const getPrototypeOf = Object.getPrototypeOf;

// function installSymbolHasInstanceImpls(context: any) {
//   nodeGlobals.forEach((name) => {
//     const Constructor = context[name];
//     if (typeof Constructor === 'function') {
//       const Prototype = Constructor.prototype;
//       Object.defineProperty(Constructor, Symbol.hasInstance, {
//         configurable: true,
//         value: (x: any) => {
//           if (x?.constructor?.name === name) { return true; }
//           for (let p = x; p != null && (p = getPrototypeOf(p));) {
//             if (p === Prototype) { return true; }
//           }
//           return false;
//         },
//       });
//     }
//   });

//   return context;
// }

// function resolve(
//   request: string, parent: Module, isMain?: boolean, options?: {paths?: string[]|undefined;}) {
//   switch (request) {
//     case 'mjolnir.js/src/utils/hammer':
//     case 'mjolnir.js/src/utils/hammer.js':
//     case 'mjolnir.js/dist/es5/utils/hammer':
//     case 'mjolnir.js/dist/es5/utils/hammer.js':
//     case 'mjolnir.js/dist/esm/utils/hammer':
//     case 'mjolnir.js/dist/esm/utils/hammer.js':
//       debugger;
//       request = request.replace('hammer', 'hammer.browser');
//       break;
//   }
//   return (Module as any)._resolveFilename(request, parent, isMain, options);
// }
