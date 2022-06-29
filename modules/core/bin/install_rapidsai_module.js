#!/usr/bin/env node

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

if (process.env.RAPIDSAI_SKIP_DOWNLOAD === '1') { return; }

const {npm_package_name: pkg} = process.env;

require('assert')(require('os').platform() === 'linux',  //
                  `${pkg} is only supported on Linux`);

const CUDA     = `11.6.2`;
const RAPIDS   = `22.06.00`;
const PKG_NAME = pkg.replace('@', '').replace('/', '_');
const GPU_ARCH = (() => {
  const cc = typeof process.env.RAPIDSAI_GPU_ARCH !== undefined
               ? [process.env.RAPIDSAI_GPU_ARCH]
               : require('@rapidsai/core').getComputeCapabilities();
  if (cc.length === 1) {
    switch (cc[0]) {
      case '60':
      case '70':
      case '75':
      case '80':
      case '86':  //
        return `${cc[0]}`;
      default: break;
    }
  }
  return '';
})();

const fs    = require('fs');
const Url   = require('url');
const Path  = require('path');
const https = require('https');
const slug  = [PKG_NAME, ...[GPU_ARCH || ``].filter(Boolean)].join('_');
const path  = Path.join(Path.dirname(require.resolve(pkg)), 'build', 'Release', `${slug}.node`);

console.log('install-rapidsai-module path:', path);

fs.access(path, fs.constants.F_OK, (err) => {
  if (err) {
    const arch = [GPU_ARCH ? `arch${GPU_ARCH}` : ``].filter(Boolean);
    const slug = [PKG_NAME, RAPIDS, `cuda${CUDA}`, `linux`, `amd64`, ...arch].join('-');

    fetch({
      hostname: `github.com`,
      path: `/rapidsai/node/releases/download/v${RAPIDS}/${slug}.node`,
      headers: {
        [`Accept`]: `application/octet-stream`,
        [`Accept-Encoding`]: `br;q=1.0, gzip;q=0.8, deflate;q=0.6, identity;q=0.4, *;q=0.1`
      }
    });

    function fetch(options = {}, numRedirects = 0) {
      https
        .get(options,
             (res) => {
               if (res.statusCode > 300 && res.statusCode < 400 && res.headers.location) {
                 const {hostname = options.hostname, path} = Url.parse(res.headers.location);
                 if (numRedirects < 10) fetch({...options, hostname, path}, numRedirects + 1);
               } else if (res.statusCode > 199 && res.statusCode < 300) {
                 const encoding = res.headers['content-encoding'] || '';
                 if (encoding.includes('gzip')) {
                   res = res.pipe(require('zlib').createGunzip());
                 } else if (encoding.includes('deflate')) {
                   res = res.pipe(require('zlib').createInflate());
                 } else if (encoding.includes('br')) {
                   res = res.pipe(require('zlib').createBrotliDecompress());
                 }
                 require('stream').pipeline(res, fs.createWriteStream(path), (e) => {});
               } else {
                 res.on('error', (e) => {}).destroy();
               }
             })
        .on('error', (e) => {})
        .end();
    }
  }
});
