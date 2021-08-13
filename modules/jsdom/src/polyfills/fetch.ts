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

import * as xfetch from 'cross-fetch';
import * as jsdom from 'jsdom';
import * as Url from 'url';

export function installFetch(window: jsdom.DOMWindow) {
  window.Headers  = xfetch.Headers;
  window.Request  = xfetch.Request;
  window.Response = xfetch.Response;
  window.fetch    = function fileAwareFetch(url: string, options: jsdom.FetchOptions) {
    const isDataURI  = url && url.startsWith('data:');
    const isFilePath = !isDataURI && !Url.parse(url).protocol;
    return !isFilePath ? xfetch.fetch(url, options)
                       // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                          : new jsdom
                           .ResourceLoader()                  //
                           .fetch(`file://${url}`, options)!  //
                           .then((x) => new Response(x, {status: 200}));
  };
  return window;
}
