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

const axios = require('axios').default;
const fs    = require('fs');
const zlib  = require('zlib');

module.exports = DownloadSQLClusterServerDataSet;

if (require.main === module) {  //
  module.exports().catch((e) => console.error(e) || process.exit(1));
}

async function DownloadSQLClusterServerDataSet() {
  if (!fs.existsSync(`${__dirname}/data`)) { fs.mkdirSync(`${__dirname}/data`); }

  console.log('Downloading dataset...');
  for (let i = 0; i < 10; ++i) { await DownloadChunk(i); }
}

async function DownloadChunk(index) {
  await axios
    .get(`https://node-rapids-data.s3.us-west-2.amazonaws.com/wikipedia/page_titles_en_${
           index}.csv.gz`,
         {responseType: 'stream'})
    .then(function(response) {
      response.data.pipe(zlib.createGunzip())
        .pipe(fs.createWriteStream(`${__dirname}/data/wiki_page_${index}.csv`));
    })
    .catch(function(error) { console.log(error); })
    .then()
}
