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

#include "node_cuda/memory.hpp"
#include "node_cuda/macros.hpp"
#include "node_cuda/utilities/napi_to_cpp.hpp"

namespace nv {

namespace {

void cudaMemsetNapi(CallbackArgs const& args) {
  Span<char> target = args[0];
  int32_t value     = args[1];
  size_t count      = args[2];
  if (args.Length() == 3) {
    NODE_CUDA_TRY(cudaMemset(target.data(), value, count));
  } else {
    cudaStream_t stream = args[3];
    NODE_CUDA_TRY(cudaMemsetAsync(target.data(), value, count, stream));
  }
}

void cudaMemcpyNapi(CallbackArgs const& args) {
  Span<char> target = args[0];
  Span<char> source = args[1];
  size_t count      = args[2];
  if (args.Length() == 3) {
    NODE_CUDA_TRY(cudaMemcpy(target.data(), source.data(), count, cudaMemcpyDefault));
  } else {
    cudaStream_t stream = args[3];
    NODE_CUDA_TRY(cudaMemcpyAsync(target.data(), source.data(), count, cudaMemcpyDefault, stream));
  }
}

}  // namespace

namespace memory {
Napi::Object initModule(Napi::Env env, Napi::Object exports) {
  EXPORT_FUNC(env, exports, "cudaMemset", cudaMemsetNapi);
  EXPORT_FUNC(env, exports, "cudaMemcpy", cudaMemcpyNapi);

  return exports;
}
}  // namespace memory

}  // namespace nv
