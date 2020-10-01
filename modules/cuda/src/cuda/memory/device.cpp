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

#include "cuda/memory.hpp"

#include <cuda_runtime_api.h>
#include <napi.h>

namespace nv {

Napi::FunctionReference DeviceMemory::constructor;

Napi::Object DeviceMemory::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor = DefineClass(
    env,
    "DeviceMemory",
    {
      InstanceAccessor("byteLength", &DeviceMemory::GetByteLength, nullptr, napi_enumerable),
      InstanceAccessor("device", &DeviceMemory::GetDevice, nullptr, napi_enumerable),
      InstanceAccessor("ptr", &DeviceMemory::GetPointer, nullptr, napi_enumerable),
      InstanceMethod("slice", &DeviceMemory::CopySlice),
    });
  DeviceMemory::constructor = Napi::Persistent(ctor);
  DeviceMemory::constructor.SuppressDestruct();

  exports.Set("DeviceMemory", ctor);

  return exports;
}

DeviceMemory::DeviceMemory(CallbackArgs const& args)
  : Napi::ObjectWrap<DeviceMemory>(args), Memory(args) {
  if (args.Length() == 1) { Initialize(args[0]); }
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
}

Napi::Value DeviceMemory::CopySlice(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  int64_t offset = args[0];
  int64_t size   = size_ - offset;
  if (args.Length() == 2 && args[1].IsNumber()) { size = args[1].operator int64_t() - offset; }
  auto copy = DeviceMemory::New(size = std::max<int64_t>(size, 0));
  if (size > 0) {
    NODE_CUDA_TRY(
      cudaMemcpy(DeviceMemory::Unwrap(copy)->base(), base() + offset, size, cudaMemcpyDefault));
  }
  return copy;
}

}  // namespace nv
