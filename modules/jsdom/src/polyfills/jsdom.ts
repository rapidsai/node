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
import * as Module from 'module';

import {
  createContextRequire as createRequire,
  CreateContextRequireOptions
} from './modules/require';
import {createContextFactory} from './modules/vm';

declare module 'jsdom' {
  // clang-format off
  interface DOMWindow {
    import: (specifier: string) => Promise<import('vm').Module>;
    evalFn: (script: () => any, globals?: Record<string, any>) => any;
  }
  // clang-format on
}

export function installJSDOMUtils(
  options: Pick<CreateContextRequireOptions, 'dir'|'resolve'|'transform'|'extensions'>) {
  return (window: jsdom.DOMWindow) => {
    const createContext = createContextFactory(window, options.dir);

    const require = window.jsdom.global.require =
      createRequire({...options, context: createContext()});

    window.import = window.jsdom.global.import = require.main._cachedDynamicImporter;

    if (window.jsdom.utils.implForWrapper(window.document)._origin === 'null') {
      window.jsdom.utils.implForWrapper(window.document)._origin = '';
    }

    window.evalFn = (f: () => any, globals: Record<string, any> = {}) => {
      const exports  = {};
      const context  = createContext(globals);
      const source   = `return (${f.toString()}).call(this);`;
      const filename = `evalmachine.<${f.name || 'anonymous'}>`;
      return require.main
        .exec(require, Module.wrap(source), filename, context)  //
        .call(exports, exports, require, require.main, filename, '.');
    };

    return window;
  };
}
