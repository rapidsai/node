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

Napi::Function ManagedMemory::Init(Napi::Env const& env, Napi::Object exports) {
  return DefineClass(
    env,
    "ManagedMemory",
    {
      InstanceValue(Napi::Symbol::WellKnown(env, "toStringTag"),
                    Napi::String::New(env, "ManagedMemory"),
                    napi_enumerable),
      InstanceAccessor("byteLength", &ManagedMemory::size, nullptr, napi_enumerable),
      InstanceAccessor("device", &ManagedMemory::device, nullptr, napi_enumerable),
      InstanceAccessor("ptr", &ManagedMemory::ptr, nullptr, napi_enumerable),
      InstanceMethod("slice", &ManagedMemory::slice),
    });
}

ManagedMemory::ManagedMemory(CallbackArgs const& args)
  : EnvLocalObjectWrap<ManagedMemory>(args), Memory(args) {
  NODE_CUDA_EXPECT(args.IsConstructCall(), "ManagedMemory constructor requires 'new'", args.Env());
  NODE_CUDA_EXPECT(args.Length() == 0 || (args.Length() == 1 && args[0].IsNumber()),
                   "ManagedMemory constructor requires a numeric byteLength argument",
                   args.Env());
  size_ = args[0];
  if (size_ > 0) {
    NODE_CUDA_TRY(cudaMallocManaged(&data_, size_));
    Napi::MemoryManagement::AdjustExternalMemory(Env(), size_);
  }
}

ManagedMemory::wrapper_t ManagedMemory::New(Napi::Env const& env, size_t size) {
  return EnvLocalObjectWrap<ManagedMemory>::New(env, size);
}

void ManagedMemory::Finalize(Napi::Env env) {
  if (data_ != nullptr && size_ > 0) {
    if (cudaFree(data_) == cudaSuccess) {
      Napi::MemoryManagement::AdjustExternalMemory(env, -size_);
    }
  }
  data_ = nullptr;
  size_ = 0;
}

Napi::Value ManagedMemory::slice(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  int64_t lhs        = args.Length() > 0 ? args[0] : 0;
  int64_t rhs        = args.Length() > 1 ? args[1] : size_;
  std::tie(lhs, rhs) = clamp_slice_args(size_, lhs, rhs);
  auto copy          = ManagedMemory::New(info.Env(), rhs - lhs);
  if (rhs - lhs > 0) {
    NODE_CUDA_TRY(cudaMemcpy(copy->base(), base() + lhs, rhs - lhs, cudaMemcpyDefault));
  }
  return copy;
}

}  // namespace nv
