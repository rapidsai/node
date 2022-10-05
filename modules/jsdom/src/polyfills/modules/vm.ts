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

import * as jsdom from 'jsdom';
import * as vm from 'vm';

const {
  implSymbol,
  implForWrapper,
  wrapperForImpl,
} = require('jsdom/lib/jsdom/living/generated/utils');

export function createContextFactory(window: jsdom.DOMWindow, cwd: string) {
  window.jsdom || (window.jsdom = {});
  window.jsdom.utils || (window.jsdom.utils = {});
  window.jsdom.utils.implSymbol     = implSymbol;
  window.jsdom.utils.implForWrapper = implForWrapper;
  window.jsdom.utils.wrapperForImpl = wrapperForImpl;
  window.jsdom.global               = implForWrapper(window)._globalObject;
  window.jsdom.global.__cwd         = cwd;
  window.jsdom.global.URL           = window.URL;
  window.jsdom.global.Blob          = window.Blob;
  window.jsdom.global.Worker        = window.Worker;

  return createContext;

  function createContext(globals: Record<string, any> = {}) {
    const outerGlobal = window.jsdom.global;
    const innerGlobal = Object.create(outerGlobal, {
      ...Object.getOwnPropertyDescriptors(global),
      ...Object.getOwnPropertyDescriptors(outerGlobal),
      global: {get: () => innerContext, configurable: true, enumerable: true},
      globalThis: {get: () => innerContext, configurable: true, enumerable: true},
      window: {value: window, configurable: true, enumerable: true, writable: false},
    });

    const innerProcess = Object.assign(clone(process), {
      __cwd: cwd,
      browser: true,
      cwd() { return this.__cwd; },
      chdir(dir: string) { this.__cwd = dir; },
    });

    const innerContext =
      vm.createContext(Object.assign(innerGlobal, {process: innerProcess}, globals));

    return installSymbolHasInstanceImpls(innerContext);
  }
}

const {
  getPrototypeOf,
  getOwnPropertyDescriptors,
} = Object;

const nodeGlobals = Object.keys(getOwnPropertyDescriptors(vm.runInNewContext('this')));

const clone = (obj: any) => Object.create(getPrototypeOf(obj), getOwnPropertyDescriptors(obj));

function installSymbolHasInstanceImpls(context: vm.Context) {
  nodeGlobals.forEach((name) => {
    const Constructor = context[name];
    if (typeof Constructor === 'function') {
      const Prototype = Constructor.prototype;
      Object.defineProperty(Constructor, Symbol.hasInstance, {
        configurable: true,
        value: (x: any) => {
          if (x?.constructor?.name === name) { return true; }
          for (let p = x; p && (p = getPrototypeOf(p));) {
            if (p === Prototype) { return true; }
          }
          return false;
        },
      });
    }
  });

  return context;
}
