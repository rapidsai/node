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

import * as gl from '@nvidia/webgl';
import * as jsdom from 'jsdom';

const {ImageData} = require('canvas');

export function installImageData(window: jsdom.DOMWindow) {
  window.ImageData ??= ImageData;
  return window;
}

export function installGetContext(window: jsdom.DOMWindow) {
  class GLFWRenderingContext extends gl.WebGL2RenderingContext {
    constructor(canvas: HTMLCanvasElement,
                window: jsdom.DOMWindow,
                options?: WebGLContextAttributes) {
      super(options);
      this.canvas                   = canvas;
      this.window                   = window;
      this.window._inputEventTarget = canvas;
    }
    private readonly window: jsdom.DOMWindow;
    public readonly canvas: HTMLCanvasElement;
    public get drawingBufferWidth() {  //
      return this.window.frameBufferWidth ?? this.window.outerWidth;
    }
    public get drawingBufferHeight() {
      return this.window.frameBufferHeight ?? this.window.outerHeight;
    }
  }

  // eslint-disable-next-line @typescript-eslint/unbound-method
  const JSDOM_getContext = window.HTMLCanvasElement.prototype.getContext;

  window.WebGLActiveInfo            = gl.WebGLActiveInfo;
  window.WebGLShaderPrecisionFormat = gl.WebGLShaderPrecisionFormat;
  window.WebGLBuffer                = gl.WebGLBuffer;
  window.WebGLContextEvent          = gl.WebGLContextEvent;
  window.WebGLFramebuffer           = gl.WebGLFramebuffer;
  window.WebGLProgram               = gl.WebGLProgram;
  window.WebGLQuery                 = gl.WebGLQuery;
  window.WebGLRenderbuffer          = gl.WebGLRenderbuffer;
  window.WebGLSampler               = gl.WebGLSampler;
  window.WebGLShader                = gl.WebGLShader;
  window.WebGLSync                  = gl.WebGLSync;
  window.WebGLTexture               = gl.WebGLTexture;
  window.WebGLTransformFeedback     = gl.WebGLTransformFeedback;
  window.WebGLUniformLocation       = gl.WebGLUniformLocation;
  window.WebGLVertexArrayObject     = gl.WebGLVertexArrayObject;
  window.WebGLRenderingContext      = GLFWRenderingContext;
  window.WebGL2RenderingContext     = GLFWRenderingContext;
  // Override Canvas's `getContext` method with one that initializes our WebGL bindings
  window.HTMLCanvasElement.prototype.getContext = getContext;

  return window;

  type RenderingContextSettings =
    CanvasRenderingContext2DSettings|ImageBitmapRenderingContextSettings|WebGLContextAttributes;

  function getContext(contextId: '2d',
                      options?: CanvasRenderingContext2DSettings): CanvasRenderingContext2D|null;

  function getContext(contextId: 'bitmaprenderer', options?: ImageBitmapRenderingContextSettings):
    ImageBitmapRenderingContext|null;

  function getContext(contextId: 'webgl', options?: WebGLContextAttributes): WebGLRenderingContext|
    null;

  function getContext(contextId: 'webgl2',
                      options?: WebGLContextAttributes): WebGL2RenderingContext|null;

  function getContext(this: any, ...args: [OffscreenRenderingContextId, RenderingContextSettings?]): RenderingContext | null {
    const [type, settings = {}] = args;
    switch (type) {
      case 'webgl':
      case 'webgl2': {
        if (!this.gl) {  //
          window._gl = this.gl = new GLFWRenderingContext(this, window, settings);
        }
        return this.gl;
      }
      default: return JSDOM_getContext.apply(this, <any>args);
    }
  }
}
