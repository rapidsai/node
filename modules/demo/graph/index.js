#!/usr/bin/env -S node --trace-uncaught

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

module.exports = (glfwOptions = {
  title: '',
  visible: true,
  transparent: false,
}) => {
  require('@babel/register')({
    cache: false,
    babelrc: false,
    cwd: __dirname,
    presets: [
      ['@babel/preset-env', {'targets': {'node': 'current'}}],
      ['@babel/preset-react', {'useBuiltIns': true}]
    ]
  });

  let args = process.argv.slice(2);
  if (args.length === 1 && args[0].includes(' ')) { args = args[0].split(' '); }

  const parseArg = (prefix, fallback = '') =>
    (args.find((arg) => arg.includes(prefix)) || `${prefix}${fallback}`).slice(prefix.length);

  const delay = Math.max(0, parseInt(parseArg('--delay=', 0)) | 0);

  async function* inputs(delay, paths) {
    const sleep = (t) => new Promise((r) => setTimeout(r, t));
    for (const path of paths.split(',')) {
      if (path) { yield path; }
      await sleep(delay);
    }
  }

  return require('@rapidsai/jsdom')
    .RapidsJSDOM.fromReactComponent(
      './src/app.js',
      {
        glfwOptions,
        // Change cwd to the example dir so relative file paths are resolved
        module: {path: __dirname},
      },
      {
        nodes: inputs(delay, parseArg('--nodes=')),
        edges: inputs(delay, parseArg('--edges=')),
        width: parseInt(parseArg('--width=', 800)) | 0,
        height: parseInt(parseArg('--height=', 600)) | 0,
        layoutParams: JSON.parse(`{${parseArg('--params=')}}`),
      });
};

if (require.main === module) {
  module.exports().window.addEventListener('close', () => process.exit(0), {once: true});
}
