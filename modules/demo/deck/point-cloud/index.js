#!/usr/bin/env -S node -r esm

// Copyright (c) 2020, NVIDIA CORPORATION.
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

const { createReactWindow } = require('@nvidia/glfw');
module.exports = createReactWindow(`${__dirname}/app.js`, true);

process.on('beforeExit', (code) => {
  debugger;
  process.stderr.write(`beforeExit (code: ${code})\n`);
});

process.on('exit', (code) => {
  debugger;
  process.stderr.write(`exit (code: ${code})\n`);
});

process.on('uncaughtException', (err, origin) => {
  debugger;
  process.stderr.write(`Uncaught Exception\n` + (origin ? `Origin: ${origin}\n` : '') +
    `Exception: ${err && err.stack || err}\n`);
});

process.on('uncaughtExceptionMonitor', (err, origin) => {
  debugger;
  process.stderr.write(`Uncaught Exception Monitor\n` + (origin ? `Origin: ${origin}\n` : '') +
    `Exception: ${err && err.stack || err}\n`);
});

process.on('rejectionHandled', (promise) => {
  debugger;
  process.stderr.write(`Handled Promise Rejection` + (promise ? `Promise: ${promise}\n` : '\n'));
});

process.on('unhandledRejection', (err, promise) => {
  debugger;
  process.stderr.write(`Unhandled Promise Rejection\n` + (promise ? `Promise: ${promise}\n` : '') +
    `Exception: ${err && err.stack || err}\n`);
});

const originalExit = process.exit;
process.exit = function () {
  debugger;
  originalExit.apply(this, arguments);
}

if (require.main === module) {
  module.exports.open({ transparent: false });
}
