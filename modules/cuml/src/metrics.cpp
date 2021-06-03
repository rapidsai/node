// Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
#include <node_cudf/utilities/napi_to_cpp.hpp>
#include <node_cuml/metrics/metrics.hpp>
#include <node_rmm/device_buffer.hpp>

#include <metrics/trustworthiness_c.h>
#include <raft/linalg/distance_type.h>
#include <raft/handle.hpp>

#include <napi.h>

namespace nv {

namespace Metrics {
Napi::Value trustworthiness(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  auto X        = DeviceBuffer::IsInstance(args[0])
                    ? reinterpret_cast<float*>(DeviceBuffer::Unwrap(args[0])->data())
                    : nullptr;
  auto embedded = DeviceBuffer::IsInstance(args[1])
                    ? reinterpret_cast<float*>(DeviceBuffer::Unwrap(args[1])->data())
                    : nullptr;
  raft::handle_t handle;
  CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

  double result = ML::Metrics::trustworthiness_score<float, raft::distance::L2SqrtUnexpanded>(
    handle, X, embedded, args[2], args[3], args[4], args[5], args[6]);
  return Napi::Value::From(info.Env(), result);
}

}  // namespace Metrics
}  // namespace nv
