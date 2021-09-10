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

const Path       = require('path');
const https      = require('https');
const fs         = require('fs');
const {finished} = require('stream/promises');
const zlib       = require('zlib');

module.exports = DownloadBlazingClusterServerDataSet;

if (require.main === module) {  //
  module.exports().catch((e) => console.error(e) || process.exit(1));
}

async function DownloadBlazingClusterServerDataSet() {
  const buffer = await DownloadDataSet();

  zlib.unzip(buffer, (err, buffer) => {
    console.log('Unzipping dataset...');
    fs.writeFileSync(Path.join(__dirname, 'wikipedia_pages.csv'), buffer.toString('utf8'));
  });
}

function DownloadDataSet() {
  console.log('Downloading dataset...');
  return new Promise((resolve, reject) => {
    const req = https.request(
      'https://node-rapids-data.s3.us-west-2.amazonaws.com/wikipedia/page_titles_en.csv.gz',
      {
        method: `GET`,
        headers: {
          [`Accept-Encoding`]: `br;q=1.0, gzip;q=0.8, deflate;q=0.6, identity;q=0.4, *;q=0.1`,
        },
      },
      (res) => {
        let body = new Uint8Array(res.headers['content-length'] | 0);

        finished(res.on('data', (part) => {
          if (body.buffer.byteLength < body.byteOffset + part.byteLength) {
            const both = new Uint8Array(
              Math.max(body.buffer.byteLength * 2, body.buffer.byteLength + part.byteLength));
            both.set(new Uint8Array(body.buffer, 0, body.byteOffset));
            body = both.subarray(body.byteOffset);
          }
          body.set(part);
          body = body.subarray(part.byteLength);
        })).then(() => { resolve(new Uint8Array(body.buffer, 0, body.byteOffset)); }).catch(reject);
      });

    req.end();

    finished(req).catch(reject);
  });
}
