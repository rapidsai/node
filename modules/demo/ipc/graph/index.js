#!/usr/bin/env -S node --experimental-vm-modules --trace-uncaught

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

module.exports = ({url, ...glfwOptions} = {
  title: 'Graph Demo',
  visible: true,
  transparent: false,
}) => {
  return require('@rapidsai/jsdom')
    .RapidsJSDOM.fromReactComponent(
      './src/app.js',
      {
        glfwOptions,
        // Change cwd to the example dir so relative file paths are resolved
        module: {path: __dirname},
      },
      {url});
};

if (require.main === module) {
  const args = process.argv.length === 3 && process.argv[2].includes(' ')
                 ? process.argv[2].split(' ')
                 : process.argv.slice(2);

  const parseArg = (prefix, fallback = '') =>
    (args.find((arg) => arg.includes(prefix)) || `${prefix}${fallback}`).slice(prefix.length);

  const url = args.find((arg) => arg.includes('tcp://')) || 'tcp://0.0.0.0:6000';

  module
    .exports({
      title: '',
      visible: true,
      transparent: false,
      url: url && require('url').parse(url),
      width: parseInt(parseArg('--width=', 800)) | 0,
      height: parseInt(parseArg('--height=', 600)) | 0,
    })
    .window.addEventListener('close', () => process.exit(0), {once: true});
}
