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

export function installWorker(window: jsdom.DOMWindow): typeof Worker {
  return class JSDOMWorker extends Worker {
    constructor(filename: string|URL, options?: WorkerOptions) {  //
      if (`${filename}`.startsWith('file:')) {
        const file = fs.readFileSync(filename, 'utf8');
        if (!file.startsWith('// rapidsai_jsdom_worker_preamble')) {
          fs.writeFileSync(filename, injectPreamble(window, file));
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
  }
}

function injectPreamble(window: jsdom.DOMWindow, rest: string) {  //
  return `// rapidsai_jsdom_worker_preamble
class ImageData {};
const {parentPort} = require('worker_threads');
class WorkerGlobalScope extends require('events') {
  constructor(global) {
    super();
    this.self = global;
    this.origin = '${window.origin}';

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
  importScripts(...xs) { xs.forEach(x => require(x)); }
  postMessage(data, ...xs) { parentPort.postMessage({data}, ...xs); }
}
global.self = new WorkerGlobalScope(Object.assign(global, {ImageData})).self;
${rest}`;
}
