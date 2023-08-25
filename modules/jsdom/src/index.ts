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

import * as jsdom from 'jsdom';

import {GLFWWindowOptions, installGLFWWindow} from './polyfills/glfw';
import {installJSDOMUtils} from './polyfills/jsdom';
import {mjolnirHammerResolvers} from './polyfills/modules/mjolnir';
import {reactMapGLMapboxResolvers} from './polyfills/modules/reactmapgl';
import {createResolve, ResolversMap} from './polyfills/modules/resolve';
import {createContextFactory} from './polyfills/modules/vm';
import {createObjectUrlAndTmpDir} from './polyfills/object-url';
import {
  AnimationFrameRequestedCallback,
  defaultFrameScheduler,
  installAnimationFrame
} from './polyfills/raf';
import {installWorker} from './polyfills/worker';
import {ImageLoader} from './resourceloader';

export interface RapidsJSDOMOptions extends jsdom.ConstructorOptions {
  module?: {path: string};
  frameRate?: number;
  glfwOptions?: GLFWWindowOptions;
  resolvers?: ResolversMap;
  babel?: false|Partial<import('@babel/core').TransformOptions>;
  reportUnhandledExceptions?: boolean;
  onAnimationFrameRequested?: AnimationFrameRequestedCallback;
}

export class RapidsJSDOM extends jsdom.JSDOM {
  static defaultOptions = {
    frameRate: 60,
    glfwOptions: {},
    reportUnhandledExceptions: true,
    onAnimationFrameRequested: undefined,
    babel: {
      babelrc: false,
      presets: [['@babel/preset-react', {'useBuiltIns': true}]],
    } as import('@babel/core').TransformOptions
  };

  static fromReactComponent(componentPath: string,
                            jsdomOptions: RapidsJSDOMOptions = {},
                            reactProps                       = {}) {
    const jsdom = new RapidsJSDOM(jsdomOptions);
    return Object.assign(jsdom, {
      loaded: jsdom.loaded.then(
        () => jsdom.window.evalFn(
          async () => {
            const {createElement} = require('react');
            const {render}        = require('react-dom');
            return await window.eval(`import('${componentPath}')`).then((Component: any) => {
              render(createElement(Component.default || Component, reactProps),
                     document.body.appendChild(document.createElement('div')));
            });
          },
          {componentPath, reactProps}))
    });
  }

  public loaded: Promise<jsdom.DOMWindow>;

  constructor(options: RapidsJSDOMOptions = {}) {
    const opts                        = Object.assign({}, RapidsJSDOM.defaultOptions, options);
    const {path: dir = process.cwd()} = opts.module ?? require.main ?? module;
    const babel                       = Object.assign(
      {}, RapidsJSDOM.defaultOptions.babel, !opts.babel ? {} : opts.babel, {cwd: dir});

    const {url, tmpdir} = createObjectUrlAndTmpDir();

    const imageLoader = new ImageLoader(url, dir);

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

        const {
          frameRate,
          glfwOptions,
          onAnimationFrameRequested = defaultFrameScheduler(window, frameRate),
        } = opts;

        window = [
          installWorker,
          installGLFWWindow(glfwOptions),
          installAnimationFrame(onAnimationFrameRequested),
          installJSDOMUtils({
            dir,
            createContext: createContextFactory(window, dir),
            resolve: createResolve({
              ...opts.resolvers,
              ...mjolnirHammerResolvers(),
              ...reactMapGLMapboxResolvers(),
            }),
          })
        ].reduce((window, fn) => fn(window), window);

        const polyfillFetchPath     = require.resolve('./polyfills/fetch');
        const polyfillImagePath     = require.resolve('./polyfills/image');
        const polyfillCanvasPath    = require.resolve('./polyfills/canvas');
        const polyfillStreamsPath   = require.resolve('./polyfills/streams');
        const polyfillObjectURLPath = require.resolve('./polyfills/object-url');
        const polyfillTransformPath = require.resolve('./polyfills/modules/transform');

        window.evalFn(() => {
          const {createTransform} =
            require(polyfillTransformPath) as typeof import('./polyfills/modules/transform');
          const {extensions: _extensions, transform: _transform} = createTransform({
            ...babel,
            preTransform(path: string, code: string) {
              // prepend a fix for mapbox-gl's serialization code
              if (path.includes('mapbox-gl/dist/mapbox-gl') ||
                  path.includes('maplibre-gl/dist/maplibre-gl')) {
                return `\
Object.defineProperty(({}).constructor, '_classRegistryKey', {value: 'Object', writable: false});
${code}`;
              }
              return code;
            }
          });

          Object.assign(window.jsdom.global.require, {extensions: _extensions});
          Object.assign(window.jsdom.global.require.main, {_extensions, _transform});

          const {installFetch} = require(polyfillFetchPath) as typeof import('./polyfills/fetch');
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
          ].reduce((window, fn) => fn(window), window);

          imageLoader.svg2img = (require('svg2img')) as typeof import('svg2img').default;
        }, {
          babel,
          tmpdir,
          frameRate,
          glfwOptions,
          imageLoader,
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
  process.on('uncaughtException' as any, (err: Error, origin: any) => {
    /* eslint-disable @typescript-eslint/restrict-template-expressions */
    process.stderr.write(`Uncaught Exception\n` + (origin ? `Origin: ${origin}\n` : '') +
                         `Exception: ${err && err.stack || err}\n`);
  });

  process.on('unhandledRejection' as any, (err: Error, promise: any) => {
    /* eslint-disable @typescript-eslint/restrict-template-expressions */
    process.stderr.write(`Unhandled Promise Rejection\n` +
                         (promise ? `Promise: ${promise}\n` : '') +
                         `Exception: ${err && err.stack || err}\n`);
  });
}
