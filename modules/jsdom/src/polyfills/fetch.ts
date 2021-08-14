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
import * as Url from 'url';

export function installFetch(window: jsdom.DOMWindow) {
  const {fetch, Headers, Request, Response} = window.evalFn(() => require('cross-fetch'));
  window.jsdom.global.Headers               = Headers;
  window.jsdom.global.Request               = Request;
  window.jsdom.global.Response              = Response;
  window.jsdom.global.fetch = function fileAwareFetch(url: string, options: jsdom.FetchOptions) {
    const isDataURI  = url && url.startsWith('data:');
    const isFilePath = !isDataURI && !Url.parse(url).protocol;
    if (isFilePath) {
      const loader  = new jsdom.ResourceLoader();
      const fileUrl = `file://localhost/${process.cwd()}/${url}`;
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
      return loader.fetch(fileUrl, options)!.then((x) => new Response(x, {status: 200}));
    }
    return fetch(url, options);
  };
  return window;
}
