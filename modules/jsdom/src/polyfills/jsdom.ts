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

export function installJSDOMUtils(window: jsdom.DOMWindow) {
  window.evalFn = (f: () => any, globals: Record<string, any> = {}) => {
    const script = new vm.Script(`(${f.toString()}).call(this)`);
    return script.runInContext(createDOMContext(window, globals));
  };
  window.jsdom || (window.jsdom = {});
  window.jsdom.utils || (window.jsdom.utils = {});
  window.jsdom.utils.implSymbol     = implSymbol;
  window.jsdom.utils.implForWrapper = implForWrapper;
  window.jsdom.utils.wrapperForImpl = wrapperForImpl;
  if (window.jsdom.utils.implForWrapper(window.document)._origin === 'null') {
    window.jsdom.utils.implForWrapper(window.document)._origin = '';
  }
  return window;
}

export function createDOMContext(window: jsdom.DOMWindow, globals: Record<string, any> = {}) {
  const context = vm.createContext([
    global,
    window,
    {global, process: {__proto__: process, browser: true}},
    globals,
  ].reduce(safeObjectAssign, {__proto__: window}));

  context.window = context;

  return context;
}

function safeObjectAssign(target: any, source: any) {
  if (source && typeof source === 'object') {
    for (const key of Object.getOwnPropertyNames(source)) {
      try {
        target[key] = source[key];
      } catch (e) { /**/
      }
    }
  }
  return target;
}
