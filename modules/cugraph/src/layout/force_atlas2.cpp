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

  float *s_positions{nullptr};
  float *x_positions{nullptr};
  float *y_positions{nullptr};

  auto positions = [&](auto const &initial_positions) mutable -> Napi::Value {
    if (initial_positions.IsObject()) {
      auto pos = initial_positions.template As<Napi::Object>();
      if (pos.Has("buffer") and pos.Get("buffer").IsObject()) {
        auto buf = pos.Get("buffer").template As<Napi::Object>();
        if (buf.Has("ptr") and buf.Get("ptr").IsNumber()) { pos = buf; }
      }
      if (pos.Has("ptr") and pos.Get("ptr").IsNumber()) {
        auto ptr    = pos.Get("ptr").template As<Napi::Number>().Int64Value();
        s_positions = reinterpret_cast<float *>(ptr);
        x_positions = s_positions;
        y_positions = x_positions + num_nodes();
      }
      return pos;
    }
    auto buf    = DeviceBuffer::New(info.Env(),
                                 std::make_unique<rmm::device_buffer>(
                                   num_nodes() * 2 * sizeof(float), rmm::cuda_stream_default, mr));
    s_positions = static_cast<float *>(buf->data());
    return buf;
  }(options.Get("positions"));

  auto graph  = this->view();
  auto handle = std::make_unique<raft::handle_t>();

  cugraph::force_atlas2(*handle,
                        graph,
                        s_positions,
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

  return positions;
}

}  // namespace nv
