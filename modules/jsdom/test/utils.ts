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

import {RapidsJSDOM} from '@rapidsai/jsdom';
import * as jsdom from 'jsdom';

export let globalWindow: jsdom.DOMWindow;

beforeAll(async () => {
  const opts = {
    ...RapidsJSDOM.defaultOptions,
    glfwOptions: {visible: false},
    module: require.main,
  };
  if (_transpileESMToCJS) {
    opts.babel.presets = [
      // transpile all ESM to CJS
      ['@babel/preset-env', {targets: {node: 'current'}}],
      ...(opts.babel.presets || []),
    ];
  }
  globalWindow = await (new RapidsJSDOM(opts)).loaded;
});

afterAll(() => {
  if (globalWindow) {  //
    globalWindow.dispatchEvent(new globalWindow.CloseEvent('close'));
  }
});

let _transpileESMToCJS = false;
export function transpileESMToCJS() { _transpileESMToCJS = true; }

export const eval_ = (code: string) => globalWindow.evalFn(() => eval(code), {code});
