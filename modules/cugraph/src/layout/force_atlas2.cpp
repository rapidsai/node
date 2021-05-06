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

#include <node_cugraph/cugraph/algorithms.hpp>
#include <node_cugraph/graph_coo.hpp>

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

Napi::Value GraphCOO::force_atlas2(Napi::CallbackInfo const &info) {
  CallbackArgs const args{info};

  NapiToCPP::Object options           = args[0];
  rmm::mr::device_memory_resource *mr = options.Get("memoryResource");

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

  float *x_start{nullptr};
  float *y_start{nullptr};

  auto positions = [&](auto const &initial_positions) mutable -> DeviceBuffer::wrapper_t {
    if (DeviceBuffer::IsInstance(initial_positions.val)) {
      auto buf = DeviceBuffer::Unwrap(initial_positions);
      x_start  = reinterpret_cast<float *>(buf->data());
      y_start  = x_start + num_nodes();
      return *buf;
    }
    return DeviceBuffer::New(info.Env(),
                             std::make_unique<rmm::device_buffer>(
                               num_nodes() * 2 * sizeof(float), rmm::cuda_stream_default, mr));
  }(options.Get("positions"));

  auto graph = this->view();

  cugraph::force_atlas2(graph,
                        reinterpret_cast<float *>(positions->data()),
                        max_iter,
                        x_start,
                        y_start,
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

  return positions;
}

}  // namespace nv
