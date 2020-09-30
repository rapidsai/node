// Copyright (c) 2020, NVIDIA CORPORATION.
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

#include "macros.hpp"
#include "utilities/cpp_to_napi.hpp"
#include "utilities/napi_to_cpp.hpp"

#include <cuda_runtime_api.h>
#include <nv_node/utilities/args.hpp>

namespace nv {

// cudaError_t cudaStreamCreate(cudaStream_t *pStream);
Napi::Value cudaStreamCreate(CallbackArgs const& info) {
  auto env = info.Env();
  cudaStream_t stream;
  CUDA_TRY(env, CUDARTAPI::cudaStreamCreate(&stream));
  return CPPToNapi(info)(stream);
}

// cudaError_t cudaStreamDestroy(cudaStream_t stream);
Napi::Value cudaStreamDestroy(CallbackArgs const& info) {
  auto env            = info.Env();
  cudaStream_t stream = info[0];
  CUDA_TRY(env, CUDARTAPI::cudaStreamDestroy(stream));
  return env.Undefined();
}

// cudaError_t cudaStreamSynchronize(cudaStream_t stream);
Napi::Value cudaStreamSynchronize(CallbackArgs const& info) {
  auto env            = info.Env();
  cudaStream_t stream = info[0];
  CUDA_TRY(env, CUDARTAPI::cudaStreamSynchronize(stream));
  return env.Undefined();
}

namespace stream {
Napi::Object initModule(Napi::Env env, Napi::Object exports) {
  EXPORT_FUNC(env, exports, "create", nv::cudaStreamCreate);
  EXPORT_FUNC(env, exports, "destroy", nv::cudaStreamDestroy);
  EXPORT_FUNC(env, exports, "synchronize", nv::cudaStreamSynchronize);

  return exports;
}
}  // namespace stream
}  // namespace nv
