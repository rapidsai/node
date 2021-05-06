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

#include "node_cuda/memory.hpp"
#include "node_cuda/utilities/napi_to_cpp.hpp"

namespace nv {

Napi::Function PinnedMemory::Init(Napi::Env const& env, Napi::Object exports) {
  return DefineClass(
    env,
    "PinnedMemory",
    {
      InstanceValue(Napi::Symbol::WellKnown(env, "toStringTag"),
                    Napi::String::New(env, "PinnedMemory"),
                    napi_enumerable),
      InstanceAccessor("byteLength", &PinnedMemory::size, nullptr, napi_enumerable),
      InstanceAccessor("device", &PinnedMemory::device, nullptr, napi_enumerable),
      InstanceAccessor("ptr", &PinnedMemory::ptr, nullptr, napi_enumerable),
      InstanceMethod("slice", &PinnedMemory::slice),
    });
}

PinnedMemory::PinnedMemory(CallbackArgs const& args)
  : EnvLocalObjectWrap<PinnedMemory>(args), Memory(args) {
  NODE_CUDA_EXPECT(args.IsConstructCall(), "PinnedMemory constructor requires 'new'", args.Env());
  NODE_CUDA_EXPECT(args.Length() == 0 || (args.Length() == 1 && args[0].IsNumber()),
                   "PinnedMemory constructor requires a numeric byteLength argument",
                   args.Env());
  size_ = args[0];
  if (size_ > 0) {
    NODE_CUDA_TRY(cudaMallocHost(&data_, size_));
    Napi::MemoryManagement::AdjustExternalMemory(Env(), size_);
  }
}

PinnedMemory::wrapper_t PinnedMemory::New(Napi::Env const& env, size_t size) {
  return EnvLocalObjectWrap<PinnedMemory>::New(env, size);
}

void PinnedMemory::Finalize(Napi::Env env) {
  if (data_ != nullptr && size_ > 0) {
    if (cudaFreeHost(data_) == cudaSuccess) {
      Napi::MemoryManagement::AdjustExternalMemory(env, -size_);
    }
  }
  data_ = nullptr;
  size_ = 0;
}

Napi::Value PinnedMemory::slice(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  int64_t lhs        = args.Length() > 0 ? args[0] : 0;
  int64_t rhs        = args.Length() > 1 ? args[1] : size_;
  std::tie(lhs, rhs) = clamp_slice_args(size_, lhs, rhs);
  auto copy          = PinnedMemory::New(info.Env(), rhs - lhs);
  if (rhs - lhs > 0) {
    NODE_CUDA_TRY(cudaMemcpy(copy->base(), base() + lhs, rhs - lhs, cudaMemcpyDefault));
  }
  return copy;
}

}  // namespace nv
