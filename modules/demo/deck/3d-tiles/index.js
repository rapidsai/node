#!/usr/bin/env -S node --trace-uncaught

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

module.exports = (glfwOptions = {
  title: '3D Titles Demo',
  transparent: false
}) => {
  return require('@rapidsai/jsdom').RapidsJSDOM.fromReactComponent('./app.js', {
    glfwOptions,
    // Change cwd to the example dir so relative file paths are resolved
    module: {path: __dirname},
  });
};

if (require.main === module) {
  module.exports().window.addEventListener('close', () => process.exit(0), {once: true});
}
