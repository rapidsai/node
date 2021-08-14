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

import * as jsdom from 'jsdom';
import * as vm from 'vm';
import {createContextRequire} from './require';

const {
  implSymbol,
  implForWrapper,
  wrapperForImpl,
} = require('jsdom/lib/jsdom/living/generated/utils');

declare module 'jsdom' {
  interface DOMWindow {
    evalFn: (script: () => any, globals?: Record<string, any>) => any;
  }
}

export function installJSDOMUtils(window: jsdom.DOMWindow, main?: NodeModule) {
  const dir = main ? main.path || require('path').dirname(main.filename || main.id) : process.cwd();

  window.evalFn = (f: () => any, globals: Record<string, any> = {}) => {
    const script = new vm.Script(`(${f.toString()}).call(this)`);
    return script.runInContext(createContext(window, globals));
  };

  window.jsdom || (window.jsdom = {});
  window.jsdom.utils || (window.jsdom.utils = {});
  window.jsdom.utils.implSymbol     = implSymbol;
  window.jsdom.utils.implForWrapper = implForWrapper;
  window.jsdom.utils.wrapperForImpl = wrapperForImpl;
  window.jsdom.global               = window.jsdom.utils.implForWrapper(window)._globalObject;

  if (window.jsdom.utils.implForWrapper(window.document)._origin === 'null') {
    window.jsdom.utils.implForWrapper(window.document)._origin = '';
  }

  return window;

  function createContext(window: jsdom.DOMWindow, globals: Record<string, any> = {}) {
    const clone = (obj: any) => Object.create(Object.getPrototypeOf(obj),  //
                                              Object.getOwnPropertyDescriptors(obj));
    const window_            = Object.assign(clone(window.jsdom.global), globals);
    const global_            = Object.create(window_, Object.getOwnPropertyDescriptors(global));
    global_.global           = global_;
    global_.process          = clone(process);
    global_.process.browser  = true;
    const context            = vm.createContext(Object.defineProperty(global_, 'window', {
      value: global_,
      writable: false,
      enumerable: true,
      configurable: true,
    }));
    context.require          = createContextRequire({context, dir});
    return context;
  }
}
