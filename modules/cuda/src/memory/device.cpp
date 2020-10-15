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
#include "node_cuda/utilities/napi_to_cpp.hpp"

namespace nv {

Napi::FunctionReference DeviceMemory::constructor;

Napi::Object DeviceMemory::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor =
    DefineClass(env,
                "DeviceMemory",
                {
                  InstanceValue(Napi::Symbol::WellKnown(env, "toStringTag"),
                                Napi::String::New(env, "DeviceMemory"),
                                napi_enumerable),
                  InstanceAccessor("byteLength", &DeviceMemory::size, nullptr, napi_enumerable),
                  InstanceAccessor("device", &DeviceMemory::device, nullptr, napi_enumerable),
                  InstanceAccessor("ptr", &DeviceMemory::ptr, nullptr, napi_enumerable),
                  InstanceMethod("slice", &DeviceMemory::slice),
                });
  DeviceMemory::constructor = Napi::Persistent(ctor);
  DeviceMemory::constructor.SuppressDestruct();

  exports.Set("DeviceMemory", ctor);

  return exports;
}

DeviceMemory::DeviceMemory(CallbackArgs const& args)
  : Napi::ObjectWrap<DeviceMemory>(args), Memory(args) {
  NODE_CUDA_EXPECT(args.IsConstructCall(), "DeviceMemory constructor requires 'new'");
  NODE_CUDA_EXPECT(args.Length() == 0 || (args.Length() == 1 && args[0].IsNumber()),
                   "DeviceMemory constructor requires a numeric byteLength argument");
  Initialize(args[0]);
}

Napi::Object DeviceMemory::New(size_t size) {
  auto inst = DeviceMemory::constructor.New({});
  DeviceMemory::Unwrap(inst)->Initialize(size);
  return inst;
}

void DeviceMemory::Initialize(size_t size) {
  size_ = size;
  if (size_ > 0) {
    NODE_CUDA_TRY(cudaMalloc(&data_, size_));
    Napi::MemoryManagement::AdjustExternalMemory(Env(), size_);
  }
}

void DeviceMemory::Finalize(Napi::Env env) {
  if (data_ != nullptr && size_ > 0) {
    if (cudaFree(data_) == cudaSuccess) {
      Napi::MemoryManagement::AdjustExternalMemory(env, -size_);
    }
  }
  data_ = nullptr;
  size_ = 0;
}

Napi::Value DeviceMemory::slice(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  int64_t lhs        = args.Length() > 0 ? args[0] : 0;
  int64_t rhs        = args.Length() > 1 ? args[1] : size_;
  std::tie(lhs, rhs) = clamp_slice_args(size_, lhs, rhs);
  auto copy          = DeviceMemory::New(rhs - lhs);
  if (rhs - lhs > 0) {
    NODE_CUDA_TRY(
      cudaMemcpy(DeviceMemory::Unwrap(copy)->base(), base() + lhs, rhs - lhs, cudaMemcpyDefault));
  }
  return copy;
}

}  // namespace nv
