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

import {installGetContext} from './polyfills/canvas';
import {installFetch} from './polyfills/fetch';
import {GLFWWindowOptions, installGLFWWindow} from './polyfills/glfw';
import {installImageData, installImageDecode} from './polyfills/image';
import {installJSDOMUtils} from './polyfills/jsdom';
import {mjolnirHammerResolvers} from './polyfills/modules/mjolnir';
import {createContextRequire as createRequire} from './polyfills/modules/require';
import {createResolve, ResolversMap} from './polyfills/modules/resolve';
import {createTransform} from './polyfills/modules/transform';
import {createContextFactory} from './polyfills/modules/vm';
import {createObjectUrlAndTmpDir} from './polyfills/object-url';
import {
  AnimationFrameRequest,
  AnimationFrameRequestedCallback,
  installAnimationFrame
} from './polyfills/raf';
import {installStreams} from './polyfills/streams';
import {ImageLoader} from './resourceloader';

export interface RapidsJSDOMOptions extends jsdom.ConstructorOptions {
  module?: {path: string};
  frameRate?: number;
  glfwOptions?: GLFWWindowOptions;
  resolvers?: ResolversMap;
  babel?: boolean|Partial<import('@babel/core').TransformOptions>;
  reportUnhandledExceptions?: boolean;
  onAnimationFrameRequested?: AnimationFrameRequestedCallback;
}

const defaultOptions = {
  glfwOptions: {},
  reportUnhandledExceptions: true,
  onAnimationFrameRequested: undefined,
  babel: {
    babelrc: false,
    presets: [
      ['@babel/preset-env', {'targets': {'node': 'current'}}],
      ['@babel/preset-react', {'useBuiltIns': true}]
    ]
  }
};

export class RapidsJSDOM extends jsdom.JSDOM {
  static fromReactComponent(componentPath: string,
                            jsdomOptions: RapidsJSDOMOptions = {},
                            reactProps                       = {}) {
    const jsdom  = new RapidsJSDOM(jsdomOptions);
    const loaded = jsdom.window.evalFn(async () => {
      const React     = require('react');
      const ReactDOM  = require('react-dom');
      const Component = await eval(`import('${componentPath}')`);
      ReactDOM.render(React.createElement(Component.default || Component, reactProps),
                      document.body.appendChild(document.createElement('div')));
    }, {componentPath, reactProps});
    return Object.assign(jsdom, {loaded});
  }

  constructor(options: RapidsJSDOMOptions = {}) {
    const opts                             = Object.assign({}, defaultOptions, options);
    const {path: dir = process.cwd()}      = opts.module ?? require.main ?? module;
    const {url, install: installObjectURL} = createObjectUrlAndTmpDir();

    const imageLoader = new ImageLoader(url, dir);

    super(undefined, {
      ...opts,
      url,
      resources: imageLoader,
      pretendToBeVisual: true,
      runScripts: 'outside-only',
      beforeParse(window) {
        if (opts.reportUnhandledExceptions) { installUnhandledExceptionListeners(); }

        const {
          onAnimationFrameRequested = defaultFrameScheduler(window, opts.frameRate),
        } = opts;

        const createContext = createContextFactory(window, dir);

        window = [
          installJSDOMUtils({
            createContext,
            require: createRequire({
              dir,
              context: createContext(),
              resolve: createResolve({...opts.resolvers, ...mjolnirHammerResolvers()}),
              ...createTransform((opts.babel || undefined) &&  //
                                     (typeof opts.babel === 'object')
                                   ? {...opts.babel, cwd: dir}
                                   : {...defaultOptions.babel, cwd: dir}),
            })
          }),
          installFetch,
          installStreams,
          installObjectURL,
          installImageData,
          installImageDecode,
          installGetContext,
          installGLFWWindow(opts.glfwOptions),
          installAnimationFrame(onAnimationFrameRequested),
        ].reduce((window, fn) => fn(window), window);

        imageLoader.svg2img = window.evalFn(() => require('svg2img').default);
      }
    });
  }
}

function defaultFrameScheduler(window: jsdom.DOMWindow, fps = 60) {
  let request: AnimationFrameRequest|null = null;
  let interval: any                       = setInterval(() => {
    if (request) {
      const f = request.flush;
      request = null;
      f(() => {});
    }
    window.poll();
  }, 1000 / fps);
  window.addEventListener('close', () => {
    interval && clearInterval(interval);
    request = interval = null;
  }, {once: true});
  return (r_: AnimationFrameRequest) => { request = r_; };
}

function installUnhandledExceptionListeners() {
  process.on(<any>'uncaughtException', (err: Error, origin: any) => {
    /* eslint-disable @typescript-eslint/restrict-template-expressions */
    process.stderr.write(`Uncaught Exception\n` + (origin ? `Origin: ${origin}\n` : '') +
                         `Exception: ${err && err.stack || err}\n`);
  });

  process.on(<any>'unhandledRejection', (err: Error, promise: any) => {
    /* eslint-disable @typescript-eslint/restrict-template-expressions */
    process.stderr.write(`Unhandled Promise Rejection\n` +
                         (promise ? `Promise: ${promise}\n` : '') +
                         `Exception: ${err && err.stack || err}\n`);
  });
}
