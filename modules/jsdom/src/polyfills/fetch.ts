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
import * as Path from 'path';
import * as Url from 'url';

export function installFetch(window: jsdom.DOMWindow) {
  const cwd: string = window.jsdom.global.__cwd;
  const {
    Headers,
    Request,
    Response,
    fetch,
  }                            = require('cross-fetch') as typeof import('cross-fetch');
  window.jsdom.global.Headers  = Headers;
  window.jsdom.global.Request  = Request;
  window.jsdom.global.Response = Response;
  window.jsdom.global.fetch    = function fileAwareFetch(inp: string|Request,
                                                      options: RequestInit = {}) {
    if (typeof inp !== 'string' && ((inp instanceof Request) || ('url' in inp))) {
      return fetch(inp, options);
    }

    let url          = inp as string;
    const isDataURI  = url && url.startsWith('data:');
    const isFilePath = url && !isDataURI && !Url.parse(url).protocol;
    if (isFilePath) {
      if (url.startsWith('/')) { url = url.slice(1); }
      const loader = new jsdom.ResourceLoader();
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
      return loader
        .fetch(`file://localhost/${cwd}/${url}`, options)!  //
        .then((x) => new Response(x, {
                status: 200,
                headers: {
                  'Content-Type': contentTypeFromPath(url),
                }
              }));
    }
    const headers = new Headers(options.headers || {});
    if (!headers.has('User-Agent')) {  //
      headers.append('User-Agent', window.navigator.userAgent);
    }
    return fetch(url, {...options, headers});
  };
  return window;
}

function contentTypeFromPath(url: string) {
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
}
