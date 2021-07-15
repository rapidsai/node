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

#include <node_cuda/utilities/error.hpp>
#include <node_cudf/utilities/buffer.hpp>
#include <node_cudf/utilities/napi_to_cpp.hpp>
#include <node_cuml/metrics.hpp>
#include <node_rmm/device_buffer.hpp>

#include <napi.h>

namespace nv {

namespace Metrics {
Napi::Value trustworthiness(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  DeviceBuffer::wrapper_t X        = data_to_devicebuffer(args.Env(), args[0], args[1]);
  DeviceBuffer::wrapper_t embedded = data_to_devicebuffer(args.Env(), args[2], args[3]);

  raft::handle_t handle;
  try {
    double result = ML::Metrics::trustworthiness_score<float, raft::distance::L2SqrtUnexpanded>(
      handle,
      static_cast<float*>(X->data()),
      static_cast<float*>(embedded->data()),
      args[4],
      args[5],
      args[6],
      args[7],
      args[8]);

    return Napi::Value::From(info.Env(), result);
  } catch (std::exception const& e) { NAPI_THROW(Napi::Error::New(info.Env(), e.what())); }
}

}  // namespace Metrics
}  // namespace nv
