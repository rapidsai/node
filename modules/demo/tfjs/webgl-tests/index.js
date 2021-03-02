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

require('@babel/register')({
    cache: false,
    babelrc: false,
    presets: [
        ["@babel/preset-env", { "targets": { "node": "current" } }],
        ['@babel/preset-react', { "useBuiltIns": true }]
    ]
});

// Open a GLFW window and run the `tfjsWebGLTests` function
require('@nvidia/glfw').createWindow(tfjsWebGLTests, true).open({
    __dirname,
    // openGLProfile: require('@nvidia/glfw').GLFWOpenGLProfile.COMPAT,
});

function tfjsWebGLTests({ __dirname }) {
    const runner = new (require('jasmine'))();
    require('@tensorflow/tfjs-core/dist/index');
    require('@tensorflow/tfjs-backend-webgl/dist/index');

    // require('@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops');
    // require('@tensorflow/tfjs-core/dist/register_all_gradients');
    // require('@tensorflow/tfjs-backend-webgl/dist/backend_webgl_test_registry');

    // Force WebGL2 and DEBUG mode
    // require('@tensorflow/tfjs-core/dist/jasmine_util').setTestEnvs([{
    //     name: 'webgl2',
    //     backendName: 'webgl',
    //     flags: {
    //         'DEBUG': true,
    //         'WEBGL_VERSION': 2,
    //         'WEBGL_CPU_FORWARD': false,
    //         'WEBGL_SIZE_UPLOAD_UNIFORM': 0
    //     },
    //     isDataSync: true
    // }]);


    require('@tensorflow/tfjs-core/dist/jasmine_util').setupTestFilters([], (testName) => {
        const toExclude = ['isBrowser: false', 'tensor in worker', 'dilation gradient'];
        for (const subStr of toExclude) {
            if (testName.includes(subStr)) { return false; }
        }
        return true;
    });

    // Import and run compiled tests copied from tfjs-backend-webgl
    require(`${__dirname}/test/backend_webgl_test`);
    // require(`${__dirname}/test/canvas_util_test`);
    // require(`${__dirname}/test/flags_webgl_test`);
    // require(`${__dirname}/test/gpgpu_context_test`);
    // require(`${__dirname}/test/gpgpu_util_test`);
    // require(`${__dirname}/test/Complex_test`);
    // require(`${__dirname}/test/Max_test`);
    // require(`${__dirname}/test/Mean_test`);
    // require(`${__dirname}/test/Reshape_test`);
    // require(`${__dirname}/test/STFT_test`);
    // require(`${__dirname}/test/reshape_packed_test`);
    // require(`${__dirname}/test/setup_test`);
    // require(`${__dirname}/test/shader_compiler_util_test`);
    // require(`${__dirname}/test/tex_util_test`);
    // require(`${__dirname}/test/webgl_batchnorm_test`);
    // require(`${__dirname}/test/webgl_custom_op_test`);
    // require(`${__dirname}/test/webgl_ops_test`);
    // require(`${__dirname}/test/webgl_topixels_test`);
    // require(`${__dirname}/test/webgl_util_test`);

    // Import and run tests from core.
    // require('@tensorflow/tfjs-core/dist/tests');

    // require('@tensorflow/tfjs-core/dist/browser_util_test');
    // require('@tensorflow/tfjs-core/dist/buffer_test');
    // require('@tensorflow/tfjs-core/dist/debug_mode_test');
    // require('@tensorflow/tfjs-core/dist/device_util_test');
    // require('@tensorflow/tfjs-core/dist/engine_test');
    // require('@tensorflow/tfjs-core/dist/environment_test');
    // require('@tensorflow/tfjs-core/dist/flags_test');
    // require('@tensorflow/tfjs-core/dist/globals_test');
    // require('@tensorflow/tfjs-core/dist/gradients_test');
    // require('@tensorflow/tfjs-core/dist/io/browser_files_test');
    // require('@tensorflow/tfjs-core/dist/io/http_test');
    // require('@tensorflow/tfjs-core/dist/io/indexed_db_test');
    // require('@tensorflow/tfjs-core/dist/io/io_utils_test');
    // require('@tensorflow/tfjs-core/dist/io/local_storage_test');
    // require('@tensorflow/tfjs-core/dist/io/model_management_test');
    // require('@tensorflow/tfjs-core/dist/io/passthrough_test');
    // require('@tensorflow/tfjs-core/dist/io/progress_test');
    // require('@tensorflow/tfjs-core/dist/io/router_registry_test');
    // require('@tensorflow/tfjs-core/dist/io/weights_loader_test');
    // require('@tensorflow/tfjs-core/dist/jasmine_util_test');
    // require('@tensorflow/tfjs-core/dist/kernel_registry_test');
    // require('@tensorflow/tfjs-core/dist/ops/abs_test');
    // require('@tensorflow/tfjs-core/dist/ops/acos_test');
    // require('@tensorflow/tfjs-core/dist/ops/acosh_test');
    // require('@tensorflow/tfjs-core/dist/ops/add_n_test');
    // require('@tensorflow/tfjs-core/dist/ops/add_test');
    // require('@tensorflow/tfjs-core/dist/ops/all_test');
    // require('@tensorflow/tfjs-core/dist/ops/any_test');
    // require('@tensorflow/tfjs-core/dist/ops/arg_max_test');
    // require('@tensorflow/tfjs-core/dist/ops/arg_min_test');
    // require('@tensorflow/tfjs-core/dist/ops/arithmetic_test');
    // require('@tensorflow/tfjs-core/dist/ops/asin_test');
    // require('@tensorflow/tfjs-core/dist/ops/asinh_test');
    // require('@tensorflow/tfjs-core/dist/ops/atan_test');
    // require('@tensorflow/tfjs-core/dist/ops/atanh_test');
    // require('@tensorflow/tfjs-core/dist/ops/avg_pool_3d_test');
    // require('@tensorflow/tfjs-core/dist/ops/avg_pool_test');
    // require('@tensorflow/tfjs-core/dist/ops/axis_util_test');
    // require('@tensorflow/tfjs-core/dist/ops/basic_lstm_cell_test');
    // require('@tensorflow/tfjs-core/dist/ops/batch_to_space_nd_test');
    // require('@tensorflow/tfjs-core/dist/ops/batchnorm_test');
    // require('@tensorflow/tfjs-core/dist/ops/binary_ops_test');
    // require('@tensorflow/tfjs-core/dist/ops/bincount_test');
    // require('@tensorflow/tfjs-core/dist/ops/boolean_mask_test');
    // require('@tensorflow/tfjs-core/dist/ops/broadcast_to_test');
    // require('@tensorflow/tfjs-core/dist/ops/broadcast_util_test');
    // require('@tensorflow/tfjs-core/dist/ops/ceil_test');
    // require('@tensorflow/tfjs-core/dist/ops/clip_by_value_test');
    // require('@tensorflow/tfjs-core/dist/ops/clone_test');
    // require('@tensorflow/tfjs-core/dist/ops/compare_ops_test');
    // require('@tensorflow/tfjs-core/dist/ops/complex_ops_test');
    // require('@tensorflow/tfjs-core/dist/ops/concat_test');
    // require('@tensorflow/tfjs-core/dist/ops/concat_util_test');
    // require('@tensorflow/tfjs-core/dist/ops/confusion_matrix_test');
    // require('@tensorflow/tfjs-core/dist/ops/conv1d_test');
    // require('@tensorflow/tfjs-core/dist/ops/conv2d_separable_test');
    // require('@tensorflow/tfjs-core/dist/ops/conv2d_test');
    // require('@tensorflow/tfjs-core/dist/ops/conv2d_transpose_test');
    // require('@tensorflow/tfjs-core/dist/ops/conv3d_test');
    // require('@tensorflow/tfjs-core/dist/ops/conv3d_transpose_test');
    // require('@tensorflow/tfjs-core/dist/ops/conv_util_test');
    // require('@tensorflow/tfjs-core/dist/ops/cos_test');
    // require('@tensorflow/tfjs-core/dist/ops/cosh_test');
    // require('@tensorflow/tfjs-core/dist/ops/cumsum_test');
    // require('@tensorflow/tfjs-core/dist/ops/dense_bincount_test');
    // require('@tensorflow/tfjs-core/dist/ops/depth_to_space_test');
    // require('@tensorflow/tfjs-core/dist/ops/depthwise_conv2d_test');
    // require('@tensorflow/tfjs-core/dist/ops/diag_test');
    // require('@tensorflow/tfjs-core/dist/ops/dilation2d_test');
    // require('@tensorflow/tfjs-core/dist/ops/dropout_test');
    // require('@tensorflow/tfjs-core/dist/ops/dropout_util_test');
    // require('@tensorflow/tfjs-core/dist/ops/elu_test');
    // require('@tensorflow/tfjs-core/dist/ops/equal_test');
    // require('@tensorflow/tfjs-core/dist/ops/erf_test');
    // require('@tensorflow/tfjs-core/dist/ops/exp_test');
    // require('@tensorflow/tfjs-core/dist/ops/expand_dims_test');
    // require('@tensorflow/tfjs-core/dist/ops/expm1_test');
    // require('@tensorflow/tfjs-core/dist/ops/eye_test');
    // require('@tensorflow/tfjs-core/dist/ops/fill_test');
    // require('@tensorflow/tfjs-core/dist/ops/floor_test');
    // require('@tensorflow/tfjs-core/dist/ops/from_pixels_test');
    // require('@tensorflow/tfjs-core/dist/ops/fused/fused_conv2d_test');
    // require('@tensorflow/tfjs-core/dist/ops/fused/fused_depthwise_conv2d_test');
    // require('@tensorflow/tfjs-core/dist/ops/fused/fused_mat_mul_test');
    // require('@tensorflow/tfjs-core/dist/ops/gather_nd_test');
    // require('@tensorflow/tfjs-core/dist/ops/gather_test');
    // require('@tensorflow/tfjs-core/dist/ops/greater_equal_test');
    // require('@tensorflow/tfjs-core/dist/ops/greater_test');
    // require('@tensorflow/tfjs-core/dist/ops/ifft_test');
    // require('@tensorflow/tfjs-core/dist/ops/image/crop_and_resize_test');
    // require('@tensorflow/tfjs-core/dist/ops/image/flip_left_right_test');
    // require('@tensorflow/tfjs-core/dist/ops/image/non_max_suppression_async_test');
    // require('@tensorflow/tfjs-core/dist/ops/image/non_max_suppression_test');
    // require('@tensorflow/tfjs-core/dist/ops/image/resize_bilinear_test');
    // require('@tensorflow/tfjs-core/dist/ops/image/resize_nearest_neighbor_test');
    // require('@tensorflow/tfjs-core/dist/ops/image/rotate_with_offset_test');
    // require('@tensorflow/tfjs-core/dist/ops/in_top_k_test');
    // require('@tensorflow/tfjs-core/dist/ops/is_finite_test');
    // require('@tensorflow/tfjs-core/dist/ops/is_inf_test');
    // require('@tensorflow/tfjs-core/dist/ops/is_nan_test');
    // require('@tensorflow/tfjs-core/dist/ops/leaky_relu_test');
    // require('@tensorflow/tfjs-core/dist/ops/less_equal_test');
    // require('@tensorflow/tfjs-core/dist/ops/less_test');
    // require('@tensorflow/tfjs-core/dist/ops/linalg/band_part_test');
    // require('@tensorflow/tfjs-core/dist/ops/linalg/gram_schmidt_test');
    // require('@tensorflow/tfjs-core/dist/ops/linalg/qr_test');
    // require('@tensorflow/tfjs-core/dist/ops/linspace_test');
    // require('@tensorflow/tfjs-core/dist/ops/local_response_normalization_test');
    // require('@tensorflow/tfjs-core/dist/ops/log1p_test');
    // require('@tensorflow/tfjs-core/dist/ops/log_sigmoid_test');
    // require('@tensorflow/tfjs-core/dist/ops/log_softmax_test');
    // require('@tensorflow/tfjs-core/dist/ops/log_sum_exp_test');
    // require('@tensorflow/tfjs-core/dist/ops/log_test');
    // require('@tensorflow/tfjs-core/dist/ops/logical_and_test');
    // require('@tensorflow/tfjs-core/dist/ops/logical_not_test');
    // require('@tensorflow/tfjs-core/dist/ops/logical_or_test');
    // require('@tensorflow/tfjs-core/dist/ops/logical_xor_test');
    // require('@tensorflow/tfjs-core/dist/ops/losses/absolute_difference_test');
    // require('@tensorflow/tfjs-core/dist/ops/losses/compute_weighted_loss_test');
    // require('@tensorflow/tfjs-core/dist/ops/losses/cosine_distance_test');
    // require('@tensorflow/tfjs-core/dist/ops/losses/hinge_loss_test');
    // require('@tensorflow/tfjs-core/dist/ops/losses/huber_loss_test');
    // require('@tensorflow/tfjs-core/dist/ops/losses/log_loss_test');
    // require('@tensorflow/tfjs-core/dist/ops/losses/mean_squared_error_test');
    // require('@tensorflow/tfjs-core/dist/ops/losses/sigmoid_cross_entropy_test');
    // require('@tensorflow/tfjs-core/dist/ops/losses/softmax_cross_entropy_test');
    // require('@tensorflow/tfjs-core/dist/ops/mat_mul_test');
    // require('@tensorflow/tfjs-core/dist/ops/max_pool_3d_test');
    // require('@tensorflow/tfjs-core/dist/ops/max_pool_test');
    // require('@tensorflow/tfjs-core/dist/ops/max_pool_with_argmax_test');
    // require('@tensorflow/tfjs-core/dist/ops/max_test');
    // require('@tensorflow/tfjs-core/dist/ops/mean_test');
    // require('@tensorflow/tfjs-core/dist/ops/min_test');
    // require('@tensorflow/tfjs-core/dist/ops/mirror_pad_test');
    // require('@tensorflow/tfjs-core/dist/ops/moments_test');
    // require('@tensorflow/tfjs-core/dist/ops/moving_average_test');
    // require('@tensorflow/tfjs-core/dist/ops/multi_rnn_cell_test');
    // require('@tensorflow/tfjs-core/dist/ops/multinomial_test');
    // require('@tensorflow/tfjs-core/dist/ops/neg_test');
    // require('@tensorflow/tfjs-core/dist/ops/norm_test');
    // require('@tensorflow/tfjs-core/dist/ops/not_equal_test');
    // require('@tensorflow/tfjs-core/dist/ops/one_hot_test');
    // require('@tensorflow/tfjs-core/dist/ops/ones_like_test');
    // require('@tensorflow/tfjs-core/dist/ops/ones_test');
    // require('@tensorflow/tfjs-core/dist/ops/operation_test');
    // require('@tensorflow/tfjs-core/dist/ops/pad_test');
    // require('@tensorflow/tfjs-core/dist/ops/pool_test');
    // require('@tensorflow/tfjs-core/dist/ops/prod_test');
    // require('@tensorflow/tfjs-core/dist/ops/rand_test');
    // require('@tensorflow/tfjs-core/dist/ops/random_gamma_test');
    // require('@tensorflow/tfjs-core/dist/ops/random_normal_test');
    // require('@tensorflow/tfjs-core/dist/ops/random_uniform_test');
    // require('@tensorflow/tfjs-core/dist/ops/range_test');
    // require('@tensorflow/tfjs-core/dist/ops/reciprocal_test');
    // require('@tensorflow/tfjs-core/dist/ops/relu6_test');
    // require('@tensorflow/tfjs-core/dist/ops/relu_test');
    // require('@tensorflow/tfjs-core/dist/ops/reverse_1d_test');
    // require('@tensorflow/tfjs-core/dist/ops/reverse_2d_test');
    // require('@tensorflow/tfjs-core/dist/ops/reverse_3d_test');
    // require('@tensorflow/tfjs-core/dist/ops/reverse_4d_test');
    // require('@tensorflow/tfjs-core/dist/ops/reverse_test');
    // require('@tensorflow/tfjs-core/dist/ops/round_test');
    // require('@tensorflow/tfjs-core/dist/ops/rsqrt_test');
    // require('@tensorflow/tfjs-core/dist/ops/scatter_nd_test');
    // require('@tensorflow/tfjs-core/dist/ops/selu_test');
    // require('@tensorflow/tfjs-core/dist/ops/setdiff1d_async_test');
    // require('@tensorflow/tfjs-core/dist/ops/sigmoid_test');
    // require('@tensorflow/tfjs-core/dist/ops/sign_test');
    // require('@tensorflow/tfjs-core/dist/ops/signal/frame_test');
    // require('@tensorflow/tfjs-core/dist/ops/signal/hamming_window_test');
    // require('@tensorflow/tfjs-core/dist/ops/signal/hann_window_test');
    // require('@tensorflow/tfjs-core/dist/ops/signal/stft_test');
    // require('@tensorflow/tfjs-core/dist/ops/sin_test');
    // require('@tensorflow/tfjs-core/dist/ops/sinh_test');
    // require('@tensorflow/tfjs-core/dist/ops/slice1d_test');
    // require('@tensorflow/tfjs-core/dist/ops/slice2d_test');
    // require('@tensorflow/tfjs-core/dist/ops/slice3d_test');
    // require('@tensorflow/tfjs-core/dist/ops/slice4d_test');
    // require('@tensorflow/tfjs-core/dist/ops/slice_test');
    // require('@tensorflow/tfjs-core/dist/ops/slice_util_test');
    // require('@tensorflow/tfjs-core/dist/ops/softmax_test');
    // require('@tensorflow/tfjs-core/dist/ops/softplus_test');
    // require('@tensorflow/tfjs-core/dist/ops/space_to_batch_nd_test');
    // require('@tensorflow/tfjs-core/dist/ops/sparse_to_dense_test');
    // require('@tensorflow/tfjs-core/dist/ops/spectral/fft_test');
    // require('@tensorflow/tfjs-core/dist/ops/spectral/irfft_test');
    // require('@tensorflow/tfjs-core/dist/ops/spectral/rfft_test');
    // require('@tensorflow/tfjs-core/dist/ops/split_test');
    // require('@tensorflow/tfjs-core/dist/ops/sqrt_test');
    // require('@tensorflow/tfjs-core/dist/ops/square_test');
    // require('@tensorflow/tfjs-core/dist/ops/stack_test');
    // require('@tensorflow/tfjs-core/dist/ops/step_test');
    // require('@tensorflow/tfjs-core/dist/ops/strided_slice_test');
    // require('@tensorflow/tfjs-core/dist/ops/sub_test');
    // require('@tensorflow/tfjs-core/dist/ops/sum_test');
    // require('@tensorflow/tfjs-core/dist/ops/tan_test');
    // require('@tensorflow/tfjs-core/dist/ops/tanh_test');
    // require('@tensorflow/tfjs-core/dist/ops/tile_test');
    // require('@tensorflow/tfjs-core/dist/ops/to_pixels_test');
    // require('@tensorflow/tfjs-core/dist/ops/topk_test');
    // require('@tensorflow/tfjs-core/dist/ops/transpose_test');
    // require('@tensorflow/tfjs-core/dist/ops/truncated_normal_test');
    // require('@tensorflow/tfjs-core/dist/ops/unique_test');
    // require('@tensorflow/tfjs-core/dist/ops/unsorted_segment_sum_test');
    // require('@tensorflow/tfjs-core/dist/ops/unstack_test');
    // require('@tensorflow/tfjs-core/dist/ops/where_async_test');
    // require('@tensorflow/tfjs-core/dist/ops/where_test');
    // require('@tensorflow/tfjs-core/dist/ops/zeros_like_test');
    // require('@tensorflow/tfjs-core/dist/ops/zeros_test');
    // require('@tensorflow/tfjs-core/dist/optimizers/adadelta_optimizer_test');
    // require('@tensorflow/tfjs-core/dist/optimizers/adagrad_optimizer_test');
    // require('@tensorflow/tfjs-core/dist/optimizers/adam_optimizer_test');
    // require('@tensorflow/tfjs-core/dist/optimizers/adamax_optimizer_test');
    // require('@tensorflow/tfjs-core/dist/optimizers/momentum_optimizer_test');
    // require('@tensorflow/tfjs-core/dist/optimizers/optimizer_test');
    // require('@tensorflow/tfjs-core/dist/optimizers/rmsprop_optimizer_test');
    // require('@tensorflow/tfjs-core/dist/optimizers/sgd_optimizer_test');
    // require('@tensorflow/tfjs-core/dist/platforms/platform_browser_test');
    // require('@tensorflow/tfjs-core/dist/platforms/platform_node_test');
    // require('@tensorflow/tfjs-core/dist/profiler_test');
    // require('@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops_test');
    // require('@tensorflow/tfjs-core/dist/serialization_test');
    // require('@tensorflow/tfjs-core/dist/tape_test');
    // require('@tensorflow/tfjs-core/dist/tensor_test');
    // require('@tensorflow/tfjs-core/dist/tensor_util_test');
    // require('@tensorflow/tfjs-core/dist/test_util_test');
    // require('@tensorflow/tfjs-core/dist/types_test');
    // require('@tensorflow/tfjs-core/dist/util_test');
    // require('@tensorflow/tfjs-core/dist/variable_test');
    // require('@tensorflow/tfjs-core/dist/version_test');
    // require('@tensorflow/tfjs-core/dist/worker_node_test');
    // require('@tensorflow/tfjs-core/dist/worker_test');

    runner.execute();
}
