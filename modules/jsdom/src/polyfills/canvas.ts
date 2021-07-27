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

class GLFWRenderingContext extends gl.WebGL2RenderingContext {
  constructor(canvas: HTMLCanvasElement,
              window: jsdom.DOMWindow,
              options?: WebGLContextAttributes) {
    super(options);
    this.canvas = canvas;
    this.window = window;
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

export function installGetContext(window: jsdom.DOMWindow) {
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const JSDOM_getContext = window.HTMLCanvasElement.prototype.getContext;

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

  function getContext(this: HTMLCanvasElement, ...args: [OffscreenRenderingContextId, RenderingContextSettings?]): RenderingContext | null {
    if ((this as any)['_webgl2_ctx']) { return (this as any)['_webgl2_ctx']; }
    switch (args[0]) {
      case 'webgl':
        return ((this as any)['_webgl2_ctx'] =
                  new GLFWRenderingContext(this, window, args[1] || {}));
      case 'webgl2':
        return ((this as any)['_webgl2_ctx'] =
                  new GLFWRenderingContext(this, window, args[1] || {}));
    }
    return JSDOM_getContext.apply(this, <any>args);
  }
}
