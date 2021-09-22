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

const fs          = require('fs');
const request     = require('request');
const zlib        = require('zlib');
const cliProgress = require('cli-progress');

module.exports = DownloadBlazingClusterServerDataSet;

if (require.main === module) {  //
  module.exports().catch((e) => console.error(e) || process.exit(1));
}

async function DownloadBlazingClusterServerDataSet() {
  const downloadBar = new cliProgress.SingleBar({}, cliProgress.Presets.shades_classic);

  console.log(' Downloading dataset...');
  var receivedBytes = 0;
  request('https://node-rapids-data.s3.us-west-2.amazonaws.com/wikipedia/page_titles_en.csv.gz')
    .on('error',
        function(err) {
          downloadBar.stop();
          console.log('Error: ' + err.message);
        })
    .on('response',
        function(data) { downloadBar.start(parseInt(data.headers['content-length']), 0) })
    .on('data',
        function(chunk) {
          receivedBytes += chunk.length;
          downloadBar.update(receivedBytes);
        })
    .on('end', function() { downloadBar.stop(); })
    .pipe(zlib.createGunzip())
    .pipe(fs.createWriteStream(`${__dirname}/wikipedia_pages.csv`));
}
