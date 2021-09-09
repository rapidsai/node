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

// Silence internal TF.js "High memory usage..."
const {MathBackendWebGL} = require('@tensorflow/tfjs-backend-webgl/dist/backend_webgl.js');

Object.defineProperty(
  MathBackendWebGL.prototype, 'warnedAboutMemory', {get() { return true; }, set() { /* noop */ }});

const Jasmine = require('jasmine');
const runner  = new Jasmine();

require('@tensorflow/tfjs-core/dist/index.js');
require('@tensorflow/tfjs-backend-webgl/dist/index.js');

const {setTestEnvs, setupTestFilters} = require('@tensorflow/tfjs-core/dist/jasmine_util.js');

// Force WebGL2
setTestEnvs([{
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

setupTestFilters([], (testName) => {
  const toExclude = ['isBrowser: false', 'tensor in worker', 'dilation gradient'];
  for (const subStr of toExclude) {
    if (testName.includes(subStr)) { return false; }
  }
  return true;
});

// // Import and run tfjs-core tests
require('@tensorflow/tfjs-core/dist/tests.js');
// require('@tensorflow/tfjs-core/dist/io/io_utils_test.js');

// // Import and run tfjs-backend-webgl tests
require('./test/backend_webgl_test.js');
require('./test/canvas_util_test.js');
require('./test/flags_webgl_test.js');
require('./test/gpgpu_context_test.js');
require('./test/gpgpu_util_test.js');
require('./test/Complex_test.js');
require('./test/Max_test.js');
require('./test/Mean_test.js');
require('./test/Reshape_test.js');
require('./test/STFT_test.js');
require('./test/reshape_packed_test.js');
require('./test/shader_compiler_util_test.js');
require('./test/tex_util_test.js');
require('./test/webgl_batchnorm_test.js');
require('./test/webgl_custom_op_test.js');
require('./test/webgl_ops_test.js');
require('./test/webgl_topixels_test.js');
require('./test/webgl_util_test.js');

module.exports = runner;
