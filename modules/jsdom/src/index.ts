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
import * as Path from 'path';
import * as Url from 'url';

import {installGetContext, installImageData} from './polyfills/canvas';
import {installGLFWWindow} from './polyfills/glfw';
import {createObjectUrlAndTmpDir} from './polyfills/object-url';
import {
  AnimationFrameRequest,
  AnimationFrameRequestedCallback,
  installAnimationFrame
} from './polyfills/raf';
import {installRequire} from './polyfills/require';

export interface RapidsJSDOMOptions extends jsdom.ConstructorOptions {
  frameRate?: number;
  onAnimationFrameRequested?: AnimationFrameRequestedCallback;
}

export class RapidsJSDOM extends jsdom.JSDOM {
  constructor(html?: string, options: RapidsJSDOMOptions = {}) {
    const {installObjectURL, tmpdir} = createObjectUrlAndTmpDir();
    const url                        = `http://${Path.basename(tmpdir)}/`.toLowerCase();
    super(html, {
      ...options,
      url,
      pretendToBeVisual: true,
      runScripts: 'outside-only',
      resources: new ImageLoader(url),
      beforeParse(window) {  //
        installRequire(window);
        installObjectURL(window);
        installImageData(window);
        installGLFWWindow(window);
        installGetContext(window);
        installAnimationFrame(window,
                              options.onAnimationFrameRequested || defaultFrameScheduler(options));
      }
    });
  }
}

class ImageLoader extends jsdom.ResourceLoader {
  constructor(private _url: string) { super(); }
  fetch(url: string, options: jsdom.FetchOptions) {
    // Hack since JSDOM 16.2.2: If loading a relative file
    // from our dummy localhost URI, translate to a file:// URI.
    if (url.startsWith(this._url)) { url = url.slice(this._url.length); }
    // url.endsWith('/') && (url = url.slice(0, -1));
    const isDataURI  = url && url.startsWith('data:');
    const isFilePath = !isDataURI && !Url.parse(url).protocol;
    return super.fetch(isFilePath ? `file://${process.cwd()}/${url}` : url, options);
  }
}

function defaultFrameScheduler({frameRate: fps = 60}: RapidsJSDOMOptions) {
  let request: AnimationFrameRequest|null = null;
  setInterval(() => {
    if (request) { request.flush(() => request = null); }
  }, 1000 / fps);
  return (r: AnimationFrameRequest) => { request = r; };
}
