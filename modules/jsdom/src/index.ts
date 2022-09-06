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

import {GLFWWindowOptions} from './polyfills/glfw';
import {installJSDOMUtils} from './polyfills/jsdom';
import {mjolnirHammerResolvers} from './polyfills/modules/mjolnir';
import {reactMapGLMapboxResolvers} from './polyfills/modules/reactmapgl';
import {createResolve, ResolversMap} from './polyfills/modules/resolve';
import {createObjectUrlAndTmpDir} from './polyfills/object-url';
import {AnimationFrameRequestedCallback, defaultFrameScheduler} from './polyfills/raf';
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
  frameRate: 60,
  glfwOptions: {},
  reportUnhandledExceptions: true,
  onAnimationFrameRequested: undefined,
  babel: {
    babelrc: false,
    presets: [
      // Uncomment this if we want to transpile all ESM to CJS
      // ['@babel/preset-env', {'targets': {'node': 'current'}}],
      ['@babel/preset-react', {'useBuiltIns': true}],
    ],
  }
};

export class RapidsJSDOM extends jsdom.JSDOM {
  static fromReactComponent(componentPath: string,
                            jsdomOptions: RapidsJSDOMOptions = {},
                            reactProps                       = {}) {
    const jsdom = new RapidsJSDOM(jsdomOptions);
    return Object.assign(jsdom, {
      loaded: jsdom.window.evalFn(
        async () => {
          const {createElement} = require('react');
          const {render}        = require('react-dom');
          return window.eval(`import('${componentPath}')`).then((Component: any) => {
            render(createElement(Component.default || Component, reactProps),
                   document.body.appendChild(document.createElement('div')));
          });
        },
        {componentPath, reactProps})
    });
  }

  public loaded: Promise<jsdom.DOMWindow>;

  constructor(options: RapidsJSDOMOptions = {}) {
    const opts                        = Object.assign({}, defaultOptions, options);
    const {path: dir = process.cwd()} = opts.module ?? require.main ?? module;
    const babel = Object.assign({}, defaultOptions.babel, opts.babel, {cwd: dir});

    const {url, tmpdir} = createObjectUrlAndTmpDir();

    const imageLoader = new ImageLoader(url, dir);

    const polyfillRAFPath       = require.resolve('./polyfills/raf');
    const polyfillGLFWPath      = require.resolve('./polyfills/glfw');
    const polyfillFetchPath     = require.resolve('./polyfills/fetch');
    const polyfillImagePath     = require.resolve('./polyfills/image');
    const polyfillCanvasPath    = require.resolve('./polyfills/canvas');
    const polyfillStreamsPath   = require.resolve('./polyfills/streams');
    const polyfillObjectURLPath = require.resolve('./polyfills/object-url');
    const polyfillTransformPath = require.resolve('./polyfills/modules/transform');

    super(undefined, {
      ...opts,
      url,
      resources: imageLoader,
      pretendToBeVisual: true,
      runScripts: options.runScripts ?? 'outside-only',
      beforeParse: (window) => {
        if (opts.reportUnhandledExceptions) {  //
          installUnhandledExceptionListeners();
        }

        window = installJSDOMUtils({
          dir,
          resolve: createResolve({
            ...opts.resolvers,
            ...mjolnirHammerResolvers(),
            ...reactMapGLMapboxResolvers(),
          }),
        })(window);

        const {
          frameRate,
          glfwOptions,
          onAnimationFrameRequested = defaultFrameScheduler(window, frameRate),
        } = opts;

        window.evalFn(() => {
          const {createTransform} =
            require(polyfillTransformPath) as typeof import('./polyfills/modules/transform');

          const {extensions: _extensions, transform: _transform} = createTransform(babel);
          Object.assign(window.jsdom.global.require, {extensions: _extensions});
          Object.assign(window.jsdom.global.require.main, {_extensions, _transform});

          const {installAnimationFrame} =
            require(polyfillRAFPath) as typeof import('./polyfills/raf');
          const {installGLFWWindow} =
            require(polyfillGLFWPath) as typeof import('./polyfills/glfw');
          const {installFetch} =  //
            require(polyfillFetchPath) as typeof import('./polyfills/fetch');
          const {installImageData, installImageDecode} =
            require(polyfillImagePath) as typeof import('./polyfills/image');
          const {installGetContext} =
            require(polyfillCanvasPath) as typeof import('./polyfills/canvas');
          const {installStreams} =
            require(polyfillStreamsPath) as typeof import('./polyfills/streams');
          const {installObjectURL} =
            require(polyfillObjectURLPath) as typeof import('./polyfills/object-url');

          [installFetch,
           installStreams,
           installObjectURL(tmpdir),
           installImageData,
           installImageDecode,
           installGetContext,
           installGLFWWindow(glfwOptions),
           installAnimationFrame(onAnimationFrameRequested),
          ].reduce((window, fn) => fn(window), window);

          imageLoader.svg2img = require('svg2img').default;
        }, {
          babel,
          tmpdir,
          frameRate,
          glfwOptions,
          imageLoader,
          polyfillRAFPath,
          polyfillGLFWPath,
          polyfillFetchPath,
          polyfillImagePath,
          polyfillCanvasPath,
          polyfillStreamsPath,
          polyfillObjectURLPath,
          polyfillTransformPath,
          onAnimationFrameRequested,
        });
      }
    });

    this.loaded = Promise.resolve(this.window);
  }
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
