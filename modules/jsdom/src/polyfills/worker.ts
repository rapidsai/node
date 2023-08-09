// Copyright (c) 2022, NVIDIA CORPORATION.
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

import * as fs from 'fs';
import * as jsdom from 'jsdom';
import {Worker, WorkerOptions} from 'worker_threads';

export function installWorker(window: jsdom.DOMWindow) {
  window.Worker = class JSDOMWorker extends Worker {
    constructor(filename: string|URL, options?: WorkerOptions) {  //
      /* eslint-disable-next-line @typescript-eslint/restrict-template-expressions */
      if (`${filename}`.startsWith('file:')) {
        const contents = fs.readFileSync(filename, 'utf8');
        if (!contents.startsWith('// rapidsai_jsdom_worker_preamble')) {
          /* eslint-disable-next-line @typescript-eslint/restrict-template-expressions */
          fs.writeFileSync(filename, injectPreamble(`${filename}`, contents));
        }
      }
      super(filename, options);
    }
    addEventListener(...[type, handler]: Parameters<Worker['addListener']>) {
      return this.addListener(type, handler);
    }
    removeEventListener(...[type, handler]: Parameters<Worker['removeListener']>) {
      return this.removeListener(type, handler);
    }
  };
  return window;
}

function injectPreamble(filename: string, code: string) {
  return `// rapidsai_jsdom_worker_preamble
const Url = require('url');
const Path = require('path');
const {Blob} = require('buffer');
const syncRequest = require('${require.resolve('sync-request')}');
class ImageData {
  constructor(data, width, height, settings) {
    if(typeof data === 'number') {
      settings = height, height = width, width = data, data = undefined;
    }
    if (data) {
      if (data.byteLength === 0) throw new RangeError("The input data has a zero byte length");
      const pitch = (() => {
        if (data instanceof Uint16Array) { return 2; }
        if (data instanceof Uint8Array) { return 4; }
        if (data instanceof Uint8ClampedArray) { return 4; }
        throw new TypeError('Expected (Uint8ClampedArray, width[, height]), (Uint16Array, width[, height]) or (width, height)');
      })();
      if (typeof width !== 'number' || width !== width) throw new RangeError("The source width is zero");
      if (typeof height !== 'number' || height !== height) height = (data.byteLength / pitch) / width;
      data = new Uint8ClampedArray(data.buffer);
    } else {
      if (typeof width !== 'number' || width !== width) throw new RangeError("The source width is zero");
      if (typeof height !== 'number' || height !== height) throw new RangeError("The source height is zero");
      data = new Uint8ClampedArray(width * height * 4);
    }
    this.data = data;
    this.width = width;
    this.height = height;
  }
}
const {parentPort} = require('worker_threads');
class WorkerGlobalScope extends require('events') {
  constructor(global) {
    super();
    this.self = global;
    this.origin = '${filename}';

    if (!global.fetch) {
      const {Headers, Request, Response, fetch} = require('${require.resolve('cross-fetch')}');
      this.fetch = fetch;
      this.Headers = Headers;
      this.Request = Request;
      this.Response = Response;
    }

    const messageHandlersMap = new Map();
    this.addEventListener = (type, handler) => {
      if (type === 'message') {
        const h = (data) => { handler({data}); };
        messageHandlersMap.set(handler, h);
        parentPort.addListener(type, h);
      } else {
        this.addListener(type, handler);
      }
    }
    this.removeEventListener = (type, handler) => {
      if (type === 'message') {
        const h = messageHandlersMap.get(handler);
        messageHandlersMap.delete(handler);
        parentPort.removeListener(type, h);
      } else {
        this.removeListener(type, handler);
      }
    }

    Object.setPrototypeOf(global, this);
  }
  importScripts(...xs) {
    xs.filter(Boolean).forEach(x => {
      const isDataURI  = x && x.startsWith('data:');
      const isFilePath = x && !isDataURI && !Url.parse(x).protocol;
      if(isDataURI || isFilePath) {
        require(x);
      } else {
        eval(syncRequest('GET', x, {}).getBody('utf-8'));
      }
    });
  }
  postMessage(data, ...xs) { parentPort.postMessage({data}, ...xs); }
}
global.self = new WorkerGlobalScope(Object.assign(global, {ImageData})).self;
${code}`;
}
