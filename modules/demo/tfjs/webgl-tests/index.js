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

// Open a GLFW window and run the `tfjsWebGLTests` function
require('@nvidia/glfw').createWindow(tfjsWebGLTests, true).open({
  __dirname,
  openGLProfile: require('@nvidia/glfw').GLFWOpenGLProfile.COMPAT,
});

function tfjsWebGLTests({ __dirname }) {
  // Silence all internal TF.js warnings
  console.warn = () => { };

  // Silence internal TF.js "High memory usage..."
  Object.defineProperty(
    require('@tensorflow/tfjs-backend-webgl/dist/backend_webgl').MathBackendWebGL.prototype,
    'warnedAboutMemory',
    { get() { return true; }, set() { /* noop */ } });

  const runner = new (require('jasmine'))();
  require('@tensorflow/tfjs-core/dist/index');
  require('@tensorflow/tfjs-backend-webgl/dist/index');

  // Force WebGL2
  require('@tensorflow/tfjs-core/dist/jasmine_util').setTestEnvs([{
    name: 'webgl2',
    backendName: 'webgl',
    flags: {
      // 'DEBUG': true, // force DEBUG mode
      'WEBGL_VERSION': 2,
      // 'WEBGL_CPU_FORWARD': false,
      // 'WEBGL_SIZE_UPLOAD_UNIFORM': 0,
      // 'WEBGL_RENDER_FLOAT32_ENABLED': true,
      // 'WEBGL_CHECK_NUMERICAL_PROBLEMS': false,
    },
    isDataSync: true
  }]);

  require('@tensorflow/tfjs-core/dist/jasmine_util').setupTestFilters([], (testName) => {
    const toExclude = ['isBrowser: false', 'tensor in worker', 'dilation gradient'];
    for (const subStr of toExclude) {
      if (testName.includes(subStr)) { return false; }
    }
    return true;
  });

  // Import and run tfjs-core tests
  require('@tensorflow/tfjs-core/dist/tests');

  // Import and run tfjs-backend-webgl tests
  require(`${__dirname}/test/backend_webgl_test`);
  require(`${__dirname}/test/canvas_util_test`);
  require(`${__dirname}/test/flags_webgl_test`);
  require(`${__dirname}/test/gpgpu_context_test`);
  require(`${__dirname}/test/gpgpu_util_test`);
  require(`${__dirname}/test/Complex_test`);
  require(`${__dirname}/test/Max_test`);
  require(`${__dirname}/test/Mean_test`);
  require(`${__dirname}/test/Reshape_test`);
  require(`${__dirname}/test/STFT_test`);
  require(`${__dirname}/test/reshape_packed_test`);
  require(`${__dirname}/test/shader_compiler_util_test`);
  require(`${__dirname}/test/tex_util_test`);
  require(`${__dirname}/test/webgl_batchnorm_test`);
  require(`${__dirname}/test/webgl_custom_op_test`);
  require(`${__dirname}/test/webgl_ops_test`);
  require(`${__dirname}/test/webgl_topixels_test`);
  require(`${__dirname}/test/webgl_util_test`);

  runner.execute();
}
