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

const Path                                  = require('path');
const https                                 = require('https');
const {createWriteStream}                   = require('fs');
const {finished}                            = require('stream/promises');
const {tableToIPC, RecordBatchStreamWriter} = require('apache-arrow');
const {Series, DataFrame, Int32, Float32}   = require('@rapidsai/cudf');

module.exports = loadSpatialDataset;

if (require.main === module) {  //
  module.exports().catch((e) => console.error(e) || process.exit(1));
}

async function loadSpatialDataset() {
  const points =
    (await loadTables(1)).reduce((points, table) => points ? points.concat(table) : table, null);

  await finished(
    RecordBatchStreamWriter.writeAll(DataFrame.fromArrow(tableToIPC(points)).toArrow())
      .pipe(createWriteStream(Path.join(__dirname, 'data', `${points.length}_points.arrow`))));
}

async function loadTables(numConcurrentRequests = 4) {
  const tables = [];

  for (let i = 0, n = 13; i < n;) {
    const requests = [];
    for (let j = -1; ++j < numConcurrentRequests && ++i < n;) { requests.push(loadFile(i)); }
    tables.push(...(await Promise.all(requests)).map(reshapeFileToTable));
  }

  return tables;
}

function reshapeFileToTable(file) {
  const size  = file.byteLength / 16;
  const shape = {type: new Int32, step: 4, size};
  const data  = Series.new({type: new Int32, data: file});
  return new DataFrame({
           x: data.gather(Series.sequence({...shape, init: 0})),
           y: data.gather(Series.sequence({...shape, init: 1})),
         })
    .castAll(new Float32)
    .toArrow();
}

function loadFile(i) {
  const options = {
    method: `GET`,
    path: `/spatial/2009${i < 10 ? '0' : ''}${i}.cny.gz`,
    hostname: `node-rapids-data.s3.us-west-2.amazonaws.com`,
    headers: {
      [`Accept`]: `application/octet-stream`,
      [`Accept-Encoding`]: `br;q=1.0, gzip;q=0.8, deflate;q=0.6, identity;q=0.4, *;q=0.1`,
    },
  };

  return new Promise((resolve, reject) => {
    const req = https.request(options, (res) => {
      const encoding = res.headers['content-encoding'] || '';

      let body = new Uint8Array(res.headers['content-length'] | 0);

      if (encoding.includes('gzip')) {
        res = res.pipe(require('zlib').createGunzip());
      } else if (encoding.includes('deflate')) {
        res = res.pipe(require('zlib').createInflate());
      } else if (encoding.includes('br')) {
        res = res.pipe(require('zlib').createBrotliDecompress());
      }

      finished(res.on('data',
                      (part) => {
                        if (body.buffer.byteLength < body.byteOffset + part.byteLength) {
                          const both = new Uint8Array(Math.max(
                            body.buffer.byteLength * 2, body.buffer.byteLength + part.byteLength));
                          both.set(new Uint8Array(body.buffer, 0, body.byteOffset));
                          body = both.subarray(body.byteOffset);
                        }
                        body.set(part);
                        body = body.subarray(part.byteLength);
                      }))
        .then(() => {
          console.log(`Loaded "https://${[options.hostname, options.path].join('')}"`);
          resolve(new Uint8Array(body.buffer, 0, body.byteOffset));
        })
        .catch(reject);
    });

    req.end();

    finished(req).catch(reject);
  });
}
