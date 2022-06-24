// Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <node_cugraph/cugraph/algorithms.hpp>
#include <node_cugraph/graph.hpp>

#include <node_cudf/utilities/buffer.hpp>

#include <node_rmm/device_buffer.hpp>

namespace nv {

namespace {

int get_int(NapiToCPP const &opt, int const default_val) {
  return opt.IsNumber() ? opt.operator int() : default_val;
}

bool get_bool(NapiToCPP const &opt, bool const default_val) {
  return opt.IsBoolean() ? opt.operator bool() : default_val;
}

float get_float(NapiToCPP const &opt, float const default_val) {
  return opt.IsNumber() ? opt.operator float() : default_val;
}

}  // namespace

Napi::Value Graph::force_atlas2(Napi::CallbackInfo const &info) {
  auto env = info.Env();
  CallbackArgs const args{info};

  NapiToCPP::Object options = args[0];
  auto mr                   = MemoryResource::IsInstance(options.Get("memoryResource"))
                                ? MemoryResource::wrapper_t(options.Get("memoryResource"))
                                : MemoryResource::Current(env);

  auto max_iter              = get_int(options.Get("numIterations"), 1);
  auto outbound_attraction   = get_bool(options.Get("outboundAttraction"), true);
  auto lin_log_mode          = get_bool(options.Get("linLogMode"), false);
  auto prevent_overlapping   = get_bool(options.Get("preventOverlap"), false);
  auto edge_weight_influence = get_float(options.Get("edgeWeightInfluence"), 1.0);
  auto jitter_tolerance      = get_float(options.Get("jitterTolerance"), 1.0);
  auto barnes_hut_theta      = get_float(options.Get("barnesHutTheta"), 0.5);
  auto scaling_ratio         = get_float(options.Get("scalingRatio"), 2.0);
  auto strong_gravity_mode   = get_bool(options.Get("strongGravityMode"), false);
  auto gravity               = get_float(options.Get("gravity"), 1.0);
  auto verbose               = get_bool(options.Get("verbose"), false);

  float *x_positions{nullptr};
  float *y_positions{nullptr};
  DeviceBuffer::wrapper_t positions;

  if (options.Has("positions") && options.Get("positions").IsObject()) {
    positions = data_to_devicebuffer(
      env, options.Get("positions"), cudf::data_type{cudf::type_id::FLOAT32}, mr);
    x_positions = static_cast<float *>(positions->data());
    y_positions = static_cast<float *>(positions->data()) + num_nodes();
  } else {
    positions =
      DeviceBuffer::New(env,
                        std::make_unique<rmm::device_buffer>(
                          num_nodes() * 2 * sizeof(float), rmm::cuda_stream_default, *mr));
  }

  auto graph = this->coo_view();

  try {
    cugraph::force_atlas2({rmm::cuda_stream_default},
                          graph,
                          static_cast<float *>(positions->data()),
                          max_iter,
                          x_positions,
                          y_positions,
                          outbound_attraction,
                          lin_log_mode,
                          prevent_overlapping,
                          edge_weight_influence,
                          jitter_tolerance,
                          true,
                          barnes_hut_theta,
                          scaling_ratio,
                          strong_gravity_mode,
                          gravity,
                          verbose);
  } catch (std::exception const &e) { throw Napi::Error::New(info.Env(), e.what()); }

  return positions;
}

}  // namespace nv
