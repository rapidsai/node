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

var fs      = require('fs');
var request = require('request');
var zlib    = require('zlib');

module.exports = DownloadBlazingClusterServerDataSet;

if (require.main === module) {  //
  module.exports().catch((e) => console.error(e) || process.exit(1));
}

async function DownloadBlazingClusterServerDataSet() {
  console.log('Downloading and unzipping dataset...');
  request('https://node-rapids-data.s3.us-west-2.amazonaws.com/wikipedia/page_titles_en.csv.gz')
    .on('error', function(err) { console.log('Error: ' + err.message); })
    .on('end',
        function() { console.log('Download finished, Blazing Server Demo is ready to launch.'); })
    .pipe(zlib.createGunzip())
    .pipe(fs.createWriteStream(`${__dirname}/wikipedia_pages.csv`));
}
