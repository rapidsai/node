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

require('segfault-handler').registerHandler('./crash.log');

require('@babel/register')({
  cache: false,
  babelrc: false,
  cwd: __dirname,
  presets: [
    ['@babel/preset-env', { 'targets': { 'node': 'current' } }],
    ['@babel/preset-react', { 'useBuiltIns': true }]
  ]
});

// Change cwd to the example dir so relative file paths are resolved
process.chdir(__dirname);

const { streamSDKServer, videoStream, inputStream, inputToDOMEvent } = require('./sdk');

const { stream: video, ...videoEvents } = videoStream({ id: 'ssr-video', title: 'Video' });
const { stream: input, ...inputEvents } = inputStream({ id: 'ssr-input', title: 'Input' });
const { server, ...serverEvents } = streamSDKServer({
  useIPv6: false,
  mediaPort: 47998,
  rtspPort: 49100,
  useTcpSignaling: false,
  streams: [video, input],
  createNvstLogger: process.argv.slice(2).includes('--logger'),
});

server.start();

const run = require('./server');

setTimeout(() => {

  const logEvents = (source, subjects) => {
    Object.keys(subjects).forEach((type) => {
      subjects[type].subscribe((event) => {
        console.log({
          source,
          type,
          ...JSON.parse(JSON.stringify(event))
        })
      });
    });
  }

  logEvents('video', videoEvents);
  logEvents('input', inputEvents);
  logEvents('server', serverEvents);

  run({
    server,
    video,
    input,
    videoEvents,
    inputEvents,
    inputToDOMEvent
  });
}, 100);
