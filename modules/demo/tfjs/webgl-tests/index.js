#!/usr/bin/env -S node -r esm

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

module.exports = () => {
  const {GLFWOpenGLProfile} = require('@nvidia/glfw');
  const {RapidsJSDOM}       = require('@rapidsai/jsdom');
  const jsdom               = new RapidsJSDOM({
    // Change cwd to the example dir so relative file paths are resolved
    // module: {path: __dirname},
    module,
    glfwOptions: {openGLProfile: GLFWOpenGLProfile.COMPAT}
  });

  jsdom.window.evalFn(() => {
    __babel({
      cache: false,
      babelrc: false,
      cwd: process.cwd(),
      presets: [
        ['@babel/preset-env', {'targets': {'node': 'current'}}],
        ['@babel/preset-react', {'useBuiltIns': true}]
      ]
    });

    // Silence all internal TF.js warnings
    console.warn = () => {};

    return require('./test').execute();
  }, {__babel: require('@babel/register')});

  return jsdom;
};

if (require.main === module) {
  // ensure tests are run with headless EGL
  delete process.env.DISPLAY;
  require('segfault-handler').registerHandler('./crash.log');
  module.exports().window.addEventListener('close', () => process.exit(0), {once: true});
}
