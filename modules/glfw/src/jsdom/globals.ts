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
import {parse as parseURL} from 'url';

import {installObjectURL} from './object-url';
import {installAnimationFrame} from './raf';
import {GLFWDOMWindow, GLFWDOMWindowOptions} from './window';

// Polyfill ImageData
(<any>window).ImageData = global.ImageData = require('canvas').ImageData;

// Polyfill MessagePort
(<any>window).MessagePort = global.MessagePort =
  require('message-port-polyfill').MessagePortPolyfill;

// Polyfill MessageChannel
(<any>window).MessageChannel = global.MessageChannel =
  require('message-port-polyfill').MessageChannelPolyfill;

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
  const fileFetch = (url: string, options: jsdom.FetchOptions) => {
    const isDataURI  = url && url.startsWith('data:');
    const isFilePath = !isDataURI && !parseURL(url).protocol;
    return !isFilePath ? xfetch.fetch(url, options)
                       // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                       : new jsdom
                           .ResourceLoader()                  //
                           .fetch(`file://${url}`, options)!  //
                           .then((x) => new Response(x, {status: 200}));
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

try {
  const hammerjs = require('hammerjs');
  for (const x of ['es5', 'es6', 'esm']) {
    try {
      const b = 'mjolnir.js/dist/' + x;
      const o = require(b + '/utils/hammer-overrides');
      o.enhancePointerEventInput(hammerjs.PointerEventInput);
      o.enhanceMouseInput(hammerjs.MouseInput);
      const mjolnirHammer   = require(b + '/utils/hammer');
      mjolnirHammer.Manager = hammerjs.Manager;
      mjolnirHammer.default = hammerjs;
    } catch (e) { /**/
    }
  }
} catch (e) { /**/
}

module.exports = function(options: GLFWDOMWindowOptions) {
  (<any>window as GLFWDOMWindow).init({width: 800, height: 600, ...options});
};
