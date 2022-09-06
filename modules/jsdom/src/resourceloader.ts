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

import * as webp from '@cwasm/webp';
import * as jsdom from 'jsdom';
import * as Url from 'url';

export class ImageLoader extends jsdom.ResourceLoader {
  declare private _svg2img: typeof import('svg2img').default;
  public set svg2img(f: typeof import('svg2img').default) { this._svg2img = f; }

  constructor(private _url: string, private _cwd: string) { super(); }
  fetch(url: string, options: jsdom.FetchOptions) {
    // Hack since JSDOM 16.2.2: If loading a relative file
    // from our dummy localhost URI, translate to a file:// URI.
    if (url.startsWith(this._url)) {  //
      url = url.slice(this._url.length);
    }
    const isDataURL = url && url.startsWith('data:');
    if (isDataURL) {
      const result = this._loadDataURL(url, options);
      if (result) { return <any>result; }
    }
    const isFilePath = url && !isDataURL && !Url.parse(url).protocol;
    if (isFilePath) {  //
      if (url.startsWith('/')) { url = url.slice(1); }
      return super.fetch(`file://${this._cwd}/${url}`, options);
    }
    return super.fetch(url, options);
  }
  private _loadDataURL(url: string, options: jsdom.FetchOptions) {
    const {mediaType, encoding, contents} = parseDataURLPrefix(url);
    switch (mediaType) {
      case 'image/webp':  //
        return loadWebpDataUrl(webp, encoding, contents);
      case 'image/svg+xml':  //
        return loadSVGDataUrl(this._svg2img, encoding, contents, options);
      default: break;
    }
    return undefined;
  }
}

function parseDataURLPrefix(url: string) {
  const comma           = url.indexOf(',', 5);
  const prefix          = url.slice(5, url.indexOf(',', 5));
  let mediaType         = 'text/plain';
  let encoding          = '';
  [mediaType, encoding] = prefix.indexOf(';') ? prefix.split(';') : [prefix, ''];
  return {mediaType, encoding, prefix, contents: url.slice(comma + 1)};
}

function loadSVGDataUrl(svg2img: typeof import('svg2img').default,
                        encoding: string,
                        contents: string,
                        {element}: jsdom.FetchOptions) {
  const options = {width: element?.offsetWidth, height: element?.offsetHeight};
  const data    = (() => {
    switch (encoding) {
      case 'base64':  //
        return Buffer.from(contents).toString('base64');
      default: return decodeURIComponent(contents).trim();
    }
  })();
  return new Promise<Buffer>((resolve, reject) => {
    svg2img(data, options, (err, data: Buffer) => {  //
      err == null ? resolve(data) : reject(err);
    });
  });
}

function loadWebpDataUrl(webp: typeof import('@cwasm/webp'), encoding: string, contents: string) {
  const data = (() => {
    switch (encoding) {
      case 'base64':  //
        return Buffer.from(contents, 'base64');
      default: return Buffer.from(decodeURIComponent(contents).trim());
    }
  })();
  return new Promise<ImageData>((resolve, reject) => {
    try {
      resolve(webp.decode(data));
    } catch (e) { reject(e); }
  });
}
