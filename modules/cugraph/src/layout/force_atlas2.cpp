// Copyright (c) 2021-2026, NVIDIA CORPORATION.
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

#include "node_cugraph/cugraph/algorithms.hpp"
#include "node_cugraph/graph.hpp"

#include <node_cudf/utilities/buffer.hpp>

#include <node_rmm/default_stream.hpp>
#include <node_rmm/device_buffer.hpp>

namespace nv {

namespace {

int get_int(NapiToCPP const& opt, int const default_val) {
  return opt.IsNumber() ? opt.operator int() : default_val;
}

bool get_bool(NapiToCPP const& opt, bool const default_val) {
  return opt.IsBoolean() ? opt.operator bool() : default_val;
}

float get_float(NapiToCPP const& opt, float const default_val) {
  return opt.IsNumber() ? opt.operator float() : default_val;
}

}  // namespace

Napi::Value Graph::force_atlas2(Napi::CallbackInfo const& info) {
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

  size_t positions_offset{0};
  float* x_positions{nullptr};
  float* y_positions{nullptr};

  Napi::Object positions;
  bool positions_is_device_memory_wrapper{false};

  auto is_device_memory = [&](Napi::Value const& data) -> bool {
    return data.IsObject() and                     //
           data.As<Napi::Object>().Has("ptr") and  //
           data.As<Napi::Object>().Get("ptr").IsNumber();
  };
  auto is_device_memory_wrapper = [&](Napi::Value const& data) -> bool {
    return data.IsObject() and                                   //
           data.As<Napi::Object>().Has("buffer") and             //
           data.As<Napi::Object>().Get("buffer").IsObject() and  //
           is_device_memory(data.As<Napi::Object>().Get("buffer"));
  };
  auto get_device_memory_ptr = [&](Napi::Object const& buffer) -> float* {
    return reinterpret_cast<float*>(buffer.Get("ptr").ToNumber().Int64Value()) + positions_offset;
  };

  if (options.Has("positions") && options.Get("positions").IsObject()) {
    positions = options.Get("positions");

    if (is_device_memory_wrapper(positions)) {
      positions_is_device_memory_wrapper = true;

      if (positions.Has("byteOffset")) {
        auto val = positions.Get("byteOffset");
        if (val.IsNumber()) {
          positions_offset = val.ToNumber().Int64Value();
        } else if (val.IsBigInt()) {
          bool lossless{false};
          positions_offset = val.As<Napi::BigInt>().Uint64Value(&lossless);
        }
      }

      positions = positions.Get("buffer").ToObject();
    } else if (!is_device_memory(positions)) {
      positions = data_to_devicebuffer(env, positions, cudf::data_type{cudf::type_id::FLOAT32}, mr);
    }

    x_positions = get_device_memory_ptr(positions);
    y_positions = x_positions + num_nodes();
  } else {
    positions =
      DeviceBuffer::New(env,
                        std::make_unique<rmm::device_buffer>(
                          num_nodes() * 2 * sizeof(float), nv::get_default_stream(), *mr));
  }

  auto graph = this->coo_view();

  try {
    constexpr uint64_t seed{0};
    raft::random::RngState rng_state(seed);
    raft::handle_t handle(nv::get_default_stream());
    cugraph::force_atlas2(handle,                            // handle
                          rng_state,                         // rng_state
                          graph,                             // graph
                          get_device_memory_ptr(positions),  // pos
                          max_iter,                          // max_iter
                          x_positions,                       // x_start
                          y_positions,                       // y_start
                          outbound_attraction,               // outbound_attraction_distribution
                          lin_log_mode,                      // lin_log_mode
                          prevent_overlapping,               // prevent_overlapping
                          nullptr,                           // vertex_radius_values
                          100.0,                             // overlap_scaling_ratio
                          edge_weight_influence,             // edge_weight_influence
                          jitter_tolerance,                  // jitter_tolerance
                          true,                              // barnes_hut_optimize
                          barnes_hut_theta,                  // barnes_hut_theta
                          scaling_ratio,                     // scaling_ratio
                          strong_gravity_mode,               // strong_gravity_mode
                          gravity,                           // gravity
                          nullptr,                           // vertex_mobility_values
                          nullptr,                           // vertex_mass_values
                          verbose                            // verbose
    );
  } catch (std::exception const& e) { throw Napi::Error::New(info.Env(), e.what()); }

  return positions_is_device_memory_wrapper ? options.Get("positions").As<Napi::Object>()
                                            : positions;
}

}  // namespace nv
