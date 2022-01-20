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

import {Require} from './modules/require';

declare module 'jsdom' {
  interface DOMWindow {
    evalFn: (script: () => any, globals?: Record<string, any>) => any;
  }
}

export function installJSDOMUtils({require, createContext}: {
  require: Require,
  createContext: (globals?: Record<string, any>) => vm.Context,
}) {
  return (window: jsdom.DOMWindow) => {
    window.jsdom.global.require = require;

    if (window.jsdom.utils.implForWrapper(window.document)._origin === 'null') {
      window.jsdom.utils.implForWrapper(window.document)._origin = '';
    }

    window.evalFn = (f: () => any, globals: Record<string, any> = {}) => {
      const source   = `(${f.toString()}).call(this)`;
      const filename = `evalmachine.<${f.name || 'anonymous'}>`;
      return require.main.exec(source, filename, createContext(globals));
    };

    return window;
  };
}
