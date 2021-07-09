#!/usr/bin/env node

// Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
  presets: [
    ['@babel/preset-env', { 'targets': { 'node': 'current' } }],
    ['@babel/preset-react', { 'useBuiltIns': true }]
  ]
});

// Change cwd to the example dir so relative file paths are resolved
process.chdir(__dirname);

let args = process.argv.slice(2);
if (args.length === 1 && args[0].includes(' ')) { args = args[0].split(' '); }

module.exports = require('@nvidia/glfw').createReactWindow(`${__dirname}/src/index.js`, true);

if (require.main === module) {
  const parseArg = (prefix, fallback = '') =>
    (args.find((arg) => arg.includes(prefix)) || `${prefix}${fallback}`).slice(prefix.length);

  const delay = Math.max(parseInt(parseArg('--delay=', 0)) | 0, 0);
  const url = args.find((arg) => arg.includes('tcp://')) || 'tcp://0.0.0.0:6000';

  module.exports.open({
    visible: true,
    transparent: false,
    _title: '',
    url: url && require('url').parse(url),
    nodes: inputs(delay, parseArg('--nodes=')),
    edges: inputs(delay, parseArg('--edges=')),
    width: parseInt(parseArg('--width=', 800)) | 0,
    height: parseInt(parseArg('--height=', 600)) | 0,
    layoutParams: JSON.parse(`{${parseArg('--params=')}}`),
  });

  async function* inputs(delay, paths) {
    const sleep = (t) => new Promise((r) => setTimeout(r, t));
    for (const path of paths.split(',')) {
      yield path;
      await sleep(delay);
    }
  }
}
