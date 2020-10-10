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

Napi::FunctionReference PinnedMemory::constructor;

Napi::Object PinnedMemory::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor =
    DefineClass(env,
                "PinnedMemory",
                {
                  StaticMethod("copy", &PinnedMemory::copy),
                  StaticMethod("fill", &PinnedMemory::fill),
                  InstanceAccessor("byteLength", &PinnedMemory::size, nullptr, napi_enumerable),
                  InstanceAccessor("device", &PinnedMemory::device, nullptr, napi_enumerable),
                  InstanceAccessor("ptr", &PinnedMemory::ptr, nullptr, napi_enumerable),
                  InstanceMethod("slice", &PinnedMemory::slice),
                });
  PinnedMemory::constructor = Napi::Persistent(ctor);
  PinnedMemory::constructor.SuppressDestruct();

  exports.Set("PinnedMemory", ctor);

  return exports;
}

PinnedMemory::PinnedMemory(CallbackArgs const& args)
  : Napi::ObjectWrap<PinnedMemory>(args), Memory(args) {
  if (args.Length() == 1) { Initialize(args[0]); }
}

Napi::Object PinnedMemory::New(size_t size) {
  auto inst = PinnedMemory::constructor.New({});
  PinnedMemory::Unwrap(inst)->Initialize(size);
  return inst;
}

void PinnedMemory::Initialize(size_t size) {
  size_ = size;
  if (size_ > 0) {
    NODE_CUDA_TRY(cudaMallocHost(&data_, size_));
    Napi::MemoryManagement::AdjustExternalMemory(Env(), size_);
  }
}

void PinnedMemory::Finalize(Napi::Env env) {
  if (data_ != nullptr && size_ > 0) {
    if (cudaFreeHost(data_) == cudaSuccess) {
      Napi::MemoryManagement::AdjustExternalMemory(env, -size_);
    }
  }
}

Napi::Value PinnedMemory::slice(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  int64_t offset = args[0];
  int64_t size   = size_ - offset;
  if (args.Length() == 2 && args[1].IsNumber()) { size = args[1].operator int64_t() - offset; }
  auto copy = PinnedMemory::New(size = std::max<int64_t>(size, 0));
  if (size > 0) {
    NODE_CUDA_TRY(
      cudaMemcpy(PinnedMemory::Unwrap(copy)->base(), base() + offset, size, cudaMemcpyDefault));
  }
  return copy;
}

}  // namespace nv
