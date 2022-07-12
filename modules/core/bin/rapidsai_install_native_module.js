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
try {
  require('@rapidsai/core');
} catch (e) { return; }

const {npm_package_name: pkg} = process.env;
const [...extra_files]        = process.argv.slice(2);

require('assert')(require('os').platform() === 'linux',  //
                  `${pkg} is only supported on Linux`);

const [major, minor] = process.env.npm_package_version.split('.').map(
  (x) => x.length < 2 ? new Array(2 - x.length).fill('0').join('') + x : x);

const CUDA     = `11.6.2`;
const RAPIDS   = `${major}.${minor}.00`;
const PKG_NAME = pkg.replace('@', '').replace('/', '_');
const GPU_ARCH = require('@rapidsai/core').getArchFromComputeCapabilities();

const {
  createWriteStream,
  constants: {F_OK},
}            = require('fs');
const Url    = require('url');
const Path   = require('path');
const https  = require('https');
const fs     = require('fs/promises');
const stream = require('stream/promises');
const out    = Path.join(Path.dirname(require.resolve(pkg)), 'build', 'Release');

Promise
  .all([
    [
      `${[PKG_NAME, ...[GPU_ARCH || ``].filter(Boolean)].join('_')}.node`,
      `${
          [PKG_NAME, RAPIDS, `cuda${CUDA}`, `linux`, `amd64`, GPU_ARCH ? `arch${GPU_ARCH}` : ``]
            .filter(Boolean)
            .join('-')}.node`,
    ],
    ...extra_files.map((slug) => [slug, slug])
  ].map(([localSlug, remoteSlug]) => maybeDownload(localSlug, remoteSlug)))
  .catch((e) => {
    console.error(e);
    return 1;
  })
  .then((code = 0) => process.exit(code))

function maybeDownload(localSlug, remoteSlug) {
  return new Promise((resolve, reject) => {
    const dst = Path.join(out, localSlug);
    fs.access(dst, F_OK)
      .catch(() => {
        return fs.access(out, F_OK)
          .catch(() => fs.mkdir(out, {recursive: true, mode: `0755`}))
          .then(() => fetch({
                        hostname: `github.com`,
                        path: `/rapidsai/node/releases/download/v${RAPIDS}/${remoteSlug}`,
                        headers: {
                          [`Accept`]: `application/octet-stream`,
                          [`Accept-Encoding`]:
                            `br;q=1.0, gzip;q=0.8, deflate;q=0.6, identity;q=0.4, *;q=0.1`
                        }
                      }).then((res) => stream.pipeline(res, createWriteStream(dst))));
      })
      .then(resolve, reject);
  });
}

function fetch(options = {}, numRedirects = 0) {
  return new Promise((resolve, reject) => {
    https
      .get(options,
           (res) => {
             if (res.statusCode > 300 && res.statusCode < 400 && res.headers.location) {
               const {hostname = options.hostname, path, ...rest} = Url.parse(res.headers.location);
               if (numRedirects < 10) {
                 fetch({...rest, headers: options.headers, hostname, path}, numRedirects + 1)
                   .then(resolve, reject);
               } else {
                 reject('Too many redirects');
               }
             } else if (res.statusCode > 199 && res.statusCode < 300) {
               const encoding = res.headers['content-encoding'] || '';
               if (encoding.includes('gzip')) {
                 res = res.pipe(require('zlib').createGunzip());
               } else if (encoding.includes('deflate')) {
                 res = res.pipe(require('zlib').createInflate());
               } else if (encoding.includes('br')) {
                 res = res.pipe(require('zlib').createBrotliDecompress());
               }
               resolve(res);
             } else {
               res.on('error', (e) => {}).destroy();
               reject(res.statusCode);
             }
           })
      .on('error', (e) => {})
      .end();
  });
}
