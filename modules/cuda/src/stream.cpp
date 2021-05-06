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

#include "node_cuda/utilities/cpp_to_napi.hpp"
#include "node_cuda/utilities/napi_to_cpp.hpp"

#include <cuda_runtime_api.h>
#include <nv_node/macros.hpp>
#include <nv_node/utilities/args.hpp>

namespace nv {

// cudaError_t cudaStreamCreate(cudaStream_t *pStream);
Napi::Value cudaStreamCreate(CallbackArgs const& info) {
  auto env = info.Env();
  cudaStream_t stream;
  NODE_CUDA_TRY(CUDARTAPI::cudaStreamCreate(&stream), env);
  return CPPToNapi(info)(stream);
}

// cudaError_t cudaStreamDestroy(cudaStream_t stream);
void cudaStreamDestroy(CallbackArgs const& info) {
  auto env            = info.Env();
  cudaStream_t stream = info[0];
  NODE_CUDA_TRY(CUDARTAPI::cudaStreamDestroy(stream), env);
}

// cudaError_t cudaStreamSynchronize(cudaStream_t stream);
void cudaStreamSynchronize(CallbackArgs const& info) {
  auto env            = info.Env();
  cudaStream_t stream = info[0];
  NODE_CUDA_TRY(CUDARTAPI::cudaStreamSynchronize(stream), env);
}

namespace stream {
Napi::Object initModule(Napi::Env const& env,
                        Napi::Object exports,
                        Napi::Object driver,
                        Napi::Object runtime) {
  EXPORT_FUNC(env, runtime, "create", nv::cudaStreamCreate);
  EXPORT_FUNC(env, runtime, "destroy", nv::cudaStreamDestroy);
  EXPORT_FUNC(env, runtime, "synchronize", nv::cudaStreamSynchronize);

  return exports;
}
}  // namespace stream
}  // namespace nv
