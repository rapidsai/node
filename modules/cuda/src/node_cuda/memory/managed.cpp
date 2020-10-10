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

namespace nv {

Napi::FunctionReference ManagedMemory::constructor;

Napi::Object ManagedMemory::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor =
    DefineClass(env,
                "ManagedMemory",
                {
                  StaticMethod("copy", &ManagedMemory::copy),
                  StaticMethod("fill", &ManagedMemory::fill),
                  InstanceAccessor("byteLength", &ManagedMemory::size, nullptr, napi_enumerable),
                  InstanceAccessor("device", &ManagedMemory::device, nullptr, napi_enumerable),
                  InstanceAccessor("ptr", &ManagedMemory::ptr, nullptr, napi_enumerable),
                  InstanceMethod("slice", &ManagedMemory::slice),
                });
  ManagedMemory::constructor = Napi::Persistent(ctor);
  ManagedMemory::constructor.SuppressDestruct();

  exports.Set("ManagedMemory", ctor);

  return exports;
}

ManagedMemory::ManagedMemory(CallbackArgs const& args)
  : Napi::ObjectWrap<ManagedMemory>(args), Memory(args) {
  if (args.Length() == 1) { Initialize(args[0]); }
}

Napi::Object ManagedMemory::New(size_t size) {
  auto inst = ManagedMemory::constructor.New({});
  ManagedMemory::Unwrap(inst)->Initialize(size);
  return inst;
}

void ManagedMemory::Initialize(size_t size) {
  size_ = size;
  if (size_ > 0) {
    NODE_CUDA_TRY(cudaMallocManaged(&data_, size_));
    Napi::MemoryManagement::AdjustExternalMemory(Env(), size_);
  }
}

void ManagedMemory::Finalize(Napi::Env env) {
  if (data_ != nullptr && size_ > 0) {
    if (cudaFree(data_) == cudaSuccess) {
      Napi::MemoryManagement::AdjustExternalMemory(env, -size_);
    }
  }
}

Napi::Value ManagedMemory::slice(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  int64_t offset = args[0];
  int64_t size   = size_ - offset;
  if (args.Length() == 2 && args[1].IsNumber()) { size = args[1].operator int64_t() - offset; }
  auto copy = ManagedMemory::New(size = std::max<int64_t>(size, 0));
  if (size > 0) {
    NODE_CUDA_TRY(
      cudaMemcpy(ManagedMemory::Unwrap(copy)->base(), base() + offset, size, cudaMemcpyDefault));
  }
  return copy;
}

}  // namespace nv
