// Copyright (c) 2020, NVIDIA CORPORATION.
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

import * as gl from '@nvidia/webgl';
import * as jsdom from 'jsdom';
import * as Path from 'path';
import {parse as parseURL} from 'url';

import {installObjectURL} from './object-url';
import {installAnimationFrame} from './raf';
import {GLFWDOMWindow, GLFWDOMWindowOptions} from './window';

// Polyfill ImageData
(<any>window).ImageData = global.ImageData = require('canvas').ImageData;

// TODO (ptaylor):
// These MessagePort polyfills break React v17. Not having them breaks
// mapbox-gl. React v17 is more critical to our demos at the moment,
// so they're commented out.
//
// Eventually need to figure out a polyfill that's compatible with both.

// // Polyfill MessagePort
// (<any>window).MessagePort = global.MessagePort =
//   require('message-port-polyfill').MessagePortPolyfill;

// // Polyfill MessageChannel
// (<any>window).MessageChannel = global.MessageChannel =
//   require('message-port-polyfill').MessageChannelPolyfill;

// Use node's perf_hooks for native performance.now
const {performance}       = require('perf_hooks');
(<any>window).performance = Object.create(performance);
// Polyfill the rest of the UserTiming API
(<any>window).performance = require('usertiming');
(<any>global).performance = (<any>window).performance;

class GLFWRenderingContext extends gl.WebGL2RenderingContext {
  constructor(canvas: HTMLCanvasElement, window: GLFWDOMWindow, options?: WebGLContextAttributes) {
    super(options);
    this.canvas = canvas;
    this.window = window;
    if (!this.window._inputEventTarget) { this.window._inputEventTarget = canvas; }
  }
  private readonly window: GLFWDOMWindow;
  public readonly canvas: HTMLCanvasElement;
  public get drawingBufferWidth() { return this.window.frameBufferWidth; }
  public get drawingBufferHeight() { return this.window.frameBufferHeight; }
}

// eslint-disable-next-line @typescript-eslint/unbound-method
const JSDOM_getContext = window.HTMLCanvasElement.prototype.getContext;

type RenderingContextSettings =
  CanvasRenderingContext2DSettings|ImageBitmapRenderingContextSettings|WebGLContextAttributes;

function getContext(contextId: '2d',
                    options?: CanvasRenderingContext2DSettings): CanvasRenderingContext2D|null;
function getContext(contextId: 'bitmaprenderer', options?: ImageBitmapRenderingContextSettings):
  ImageBitmapRenderingContext|null;
function getContext(contextId: 'webgl', options?: WebGLContextAttributes): WebGLRenderingContext|
  null;
function getContext(contextId: 'webgl2', options?: WebGLContextAttributes): WebGL2RenderingContext|
  null;
function getContext(this: HTMLCanvasElement, ...args: [OffscreenRenderingContextId, RenderingContextSettings?]): RenderingContext | null {
  if ((this as any)['_webgl2_ctx']) { return (this as any)['_webgl2_ctx']; }
  switch (args[0]) {
    case 'webgl':
      return ((this as any)['_webgl2_ctx'] =
                new GLFWRenderingContext(this, <any>window, args[1] || {}));
    case 'webgl2':
      return ((this as any)['_webgl2_ctx'] =
                new GLFWRenderingContext(this, <any>window, args[1] || {}));
  }
  return JSDOM_getContext.apply(this, <any>args);
}
window.WebGLActiveInfo                        = gl.WebGLActiveInfo;
window.WebGLShaderPrecisionFormat             = gl.WebGLShaderPrecisionFormat;
window.WebGLBuffer                            = gl.WebGLBuffer;
window.WebGLContextEvent                      = gl.WebGLContextEvent;
window.WebGLFramebuffer                       = gl.WebGLFramebuffer;
window.WebGLProgram                           = gl.WebGLProgram;
window.WebGLQuery                             = gl.WebGLQuery;
window.WebGLRenderbuffer                      = gl.WebGLRenderbuffer;
window.WebGLSampler                           = gl.WebGLSampler;
window.WebGLShader                            = gl.WebGLShader;
window.WebGLSync                              = gl.WebGLSync;
window.WebGLTexture                           = gl.WebGLTexture;
window.WebGLTransformFeedback                 = gl.WebGLTransformFeedback;
window.WebGLUniformLocation                   = gl.WebGLUniformLocation;
window.WebGLVertexArrayObject                 = gl.WebGLVertexArrayObject;
window.WebGLRenderingContext                  = <any>GLFWRenderingContext;
window.WebGL2RenderingContext                 = <any>GLFWRenderingContext;
window.HTMLCanvasElement.prototype.getContext = getContext;

if (!window.HTMLImageElement.prototype.decode) {
  ((window.HTMLImageElement.prototype as any).decode = function() {
    return new Promise<void>((resolve, reject) => {
      const cleanup = () => {
        this.removeEventListener('load', onload);
        this.removeEventListener('error', onerror);
      };
      const onload = () => {
        resolve();
        cleanup();
      };
      const onerror = () => {
        reject();
        cleanup();
      };
      this.addEventListener('load', onload);
      this.addEventListener('error', onerror);
    });
  });
}

Object.defineProperties(installAnimationFrame(installObjectURL(global, window)),
                        Object.getOwnPropertyDescriptors(GLFWDOMWindow.prototype));

const global_ = <any>global;

global_.performance                = window.performance;
global_.MutationObserver           = window.MutationObserver;
global_.WebGLActiveInfo            = window.WebGLActiveInfo;
global_.WebGLShaderPrecisionFormat = window.WebGLShaderPrecisionFormat;
global_.WebGLBuffer                = window.WebGLBuffer;
global_.WebGLContextEvent          = window.WebGLContextEvent;
global_.WebGLFramebuffer           = window.WebGLFramebuffer;
global_.WebGLProgram               = window.WebGLProgram;
global_.WebGLQuery                 = window.WebGLQuery;
global_.WebGLRenderbuffer          = window.WebGLRenderbuffer;
global_.WebGLSampler               = window.WebGLSampler;
global_.WebGLShader                = window.WebGLShader;
global_.WebGLSync                  = window.WebGLSync;
global_.WebGLTexture               = window.WebGLTexture;
global_.WebGLTransformFeedback     = window.WebGLTransformFeedback;
global_.WebGLUniformLocation       = window.WebGLUniformLocation;
global_.WebGLVertexArrayObject     = window.WebGLVertexArrayObject;
global_.WebGLRenderingContext      = window.WebGLRenderingContext;
global_.WebGL2RenderingContext     = window.WebGL2RenderingContext;

global_.cancelAnimationFrame  = window.cancelAnimationFrame;
global_.requestAnimationFrame = window.requestAnimationFrame;

const origin = global_.idlUtils.implForWrapper(window.document)._origin;
if (origin === 'null') { global_.idlUtils.implForWrapper(window.document)._origin = ''; }

if (typeof global_['fetch'] === 'undefined') {
  const xfetch    = require('cross-fetch');
  const fileFetch = (url: string, options: RequestInit = {}) => {
    const isDataURL  = url && url.startsWith('data:');
    const parsedUrl  = !isDataURL && url && parseURL(url);
    const isFilePath = !isDataURL && parsedUrl && !parsedUrl.protocol;
    if (isFilePath) {
      const loader  = new jsdom.ResourceLoader();
      const fileUrl = `file://localhost/${process.cwd()}/${url}`;
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
      return loader.fetch(fileUrl, options as any)!.then((x) => {  //
        const opts: ResponseInit = {
          status: 200,
          headers: {
            'Content-Type': (() => {
              switch (Path.parse(url).ext) {
                case '.jpg': return 'image/jpeg';
                case '.jpeg': return 'image/jpeg';
                case '.gif': return 'image/gif';
                case '.png': return 'image/png';
                case '.txt': return 'text/plain';
                case '.svg': return 'image/svg';
                case '.json': return 'application/json';
                case '.wasm': return 'application/wasm';
                default: return 'application/octet-stream';
              }
            })()
          }
        };
        return new xfetch.Response(x, opts);
      });
    }
    const headers: Headers = new xfetch.Headers(options.headers || {});
    if (!headers.has('User-Agent')) {  //
      headers.append('User-Agent', window.navigator.userAgent);
    }
    return xfetch.fetch(url, {...options, headers});
  };
  const xfetchDefs = {
    'fetch': {value: fileFetch, writable: true, configurable: true},
    'Response': {value: xfetch.Response, writable: true, configurable: true},
    'Headers': {value: xfetch.Headers, writable: true, configurable: true},
    'Request': {value: xfetch.Request, writable: true, configurable: true},
  };
  Object.defineProperties(global, xfetchDefs);
  Object.defineProperties(window, xfetchDefs);
}

if (typeof global_['ReadableStream'] === 'undefined') {
  const streams     = require('web-streams-polyfill');
  const streamsDefs = {
    'ReadableStream': {get() { return streams.ReadableStream; }},
    'WritableStream': {get() { return streams.WritableStream; }},
    'TransformStream': {get() { return streams.TransformStream; }},
    'CountQueuingStrategy': {get() { return streams.CountQueuingStrategy; }},
    'ByteLengthQueuingStrategy': {get() { return streams.ByteLengthQueuingStrategy; }},
  };
  Object.defineProperties(global, streamsDefs);
  Object.defineProperties(window, streamsDefs);
}

const defineLayoutProps = (elt: any) => {
  const w = window as any;
  ['width',
   'height',
   'screenY',
   'screenX',
   'screenTop',
   'screenLeft',
   'scrollTop',
   'scrollLeft',
   'pageXOffset',
   'pageYOffset',
   'clientWidth',
   'clientHeight',
   'innerWidth',
   'innerHeight',
   'offsetWidth',
   'offsetHeight',
  ].forEach((k) => Object.defineProperty(elt, k, {
    get: () => w[k],
    set: () => {},
    enumerable: true,
    configurable: true,
  }));
  elt.getBoundingClientRect = w.getBoundingClientRect.bind(window);
  return elt;
};

defineLayoutProps(window.Document.prototype);
defineLayoutProps(window.HTMLElement.prototype);

Object.defineProperties(window.SVGElement.prototype, {
  width: {get() { return {baseVal: {value: window.innerWidth}}; }},
  height: {get() { return {baseVal: {value: window.innerHeight}}; }},
});

// @ts-ignore
import * as hammerjs from 'hammerjs';
// @ts-ignore
import * as mjolnirSrcHammer from 'mjolnir.js/src/utils/hammer';
// @ts-ignore
import * as mjolnirSrcHammerOverrides from 'mjolnir.js/src/utils/hammer-overrides';

try {
  mjolnirSrcHammerOverrides.enhancePointerEventInput(hammerjs.PointerEventInput);
  mjolnirSrcHammerOverrides.enhanceMouseInput(hammerjs.MouseInput);
  Object.defineProperty(
    mjolnirSrcHammer,
    'Manager',
    {...Object.getOwnPropertyDescriptor(mjolnirSrcHammer, 'Manager'), value: hammerjs.Manager});
  Object.defineProperty(
    mjolnirSrcHammer,
    'default',
    {...Object.getOwnPropertyDescriptor(mjolnirSrcHammer, 'default'), value: hammerjs});
  for (const x of ['src', 'dist/es5', 'dist/esm']) {
    try {
      const mjolnirHammer = require(Path.join('mjolnir.js', x, 'utils/hammer'));
      Object.defineProperty(
        mjolnirHammer,
        'Manager',
        {...Object.getOwnPropertyDescriptor(mjolnirHammer, 'Manager'), value: hammerjs.Manager});
      Object.defineProperty(
        mjolnirHammer,
        'default',
        {...Object.getOwnPropertyDescriptor(mjolnirHammer, 'default'), value: hammerjs});
    } catch (e) { /**/
    }
  }
} catch (e) { /**/
}

module.exports = function(options: GLFWDOMWindowOptions) {
  (<any>window as GLFWDOMWindow).init({width: 800, height: 600, ...options});
};
