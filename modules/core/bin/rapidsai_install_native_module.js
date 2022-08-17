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

const {
  npm_package_name: pkg_name,
  npm_package_version: pkg_ver,
} = process.env;

if (process.env.RAPIDSAI_SKIP_DOWNLOAD === '1') {
  console.log(`${pkg_name}: Not downloading native module because RAPIDSAI_SKIP_DOWNLOAD=1`);
  return;
}

try {
  require('@rapidsai/core');
} catch (e) { return; }

const {
  getCudaDriverVersion,
  getArchFromComputeCapabilities,
} = require('@rapidsai/core');

require('assert')(require('os').platform() === 'linux',  //
                  `${pkg_name} is only supported on Linux`);

const {
  createWriteStream,
  constants: {F_OK},
}            = require('fs');
const Url    = require('url');
const Path   = require('path');
const https  = require('https');
const fs     = require('fs/promises');
const stream = require('stream/promises');
const getOS  = require('util').promisify(require('getos'));

const extraFiles = process.argv.slice(2);
const binary_dir = Path.join(Path.dirname(require.resolve(pkg_name)), 'build', 'Release');

(async () => {
  const distro   = await (async () => {
    const {dist = '', release = ''} = await getOS();
    return dist.toLowerCase() + release;
  })();
  const cpu_arch = (() => {
    switch (require('os').arch()) {
      case 'x64': return 'amd64';
      case 'arm': return 'aarch64';
      case 'arm64': return 'aarch64';
      default: return 'amd64';
    }
  })();
  const gpu_arch = getArchFromComputeCapabilities();
  const cuda_ver = `cuda${
    (() => {
      if (typeof process.env.RAPIDSAI_CUDA_VERSION !== 'undefined') {
        return process.env.RAPIDSAI_CUDA_VERSION;
      }
      if (typeof process.env.CUDA_VERSION_MAJOR !== 'undefined') {
        return process.env.CUDA_VERSION_MAJOR;
      }
      if (typeof process.env.CUDA_VERSION !== 'undefined') {
        return process.env.CUDA_VERSION.split('.')[0];
      }
      return getCudaDriverVersion()[0];
    })() ||
    '11'}`;
  const PKG_NAME = pkg_name.replace('@', '').replace('/', '_');
  const MOD_NAME = [PKG_NAME, pkg_ver, cuda_ver, distro, cpu_arch, gpu_arch ? `sm${gpu_arch}` : ``]
                     .filter(Boolean)
                     .join('-');

  await Promise.all([
    [
      `${PKG_NAME}.node`,
      `${MOD_NAME}.node`,
    ],
    ...extraFiles.map((slug) => [slug, slug])
  ].map(([localSlug, remoteSlug]) => maybeDownload(localSlug, remoteSlug)));
})()
  .catch((e) => {
    if (e) console.error(e);
    return 1;
  })
  .then((code = 0) => process.exit(code));

function maybeDownload(localSlug, remoteSlug) {
  return new Promise((resolve, reject) => {
    const dst = Path.join(binary_dir, localSlug);
    fs.access(dst, F_OK)
      .catch(() => {
        return fs.access(binary_dir, F_OK)
          .catch(() => fs.mkdir(binary_dir, {recursive: true, mode: `0755`}))
          .then(() => fetch({
                        hostname: `github.com`,
                        path: `/rapidsai/node/releases/download/v${pkg_ver}/${remoteSlug}`,
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
               reject({
                 statusCode: res.statusCode,
                 statusMessage: res.statusMessage,
                 url: new URL(options.path, `https://${options.hostname}`),
               });
             }
           })
      .on('error', (e) => {})
      .end();
  });
}
