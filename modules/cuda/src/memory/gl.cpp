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

Napi::FunctionReference MappedGLMemory::constructor;

Napi::Object MappedGLMemory::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor =
    DefineClass(env,
                "MappedGLMemory",
                {
                  InstanceValue(Napi::Symbol::WellKnown(env, "toStringTag"),
                                Napi::String::New(env, "MappedGLMemory"),
                                napi_enumerable),
                  InstanceAccessor("byteLength", &MappedGLMemory::size, nullptr, napi_enumerable),
                  InstanceAccessor("device", &MappedGLMemory::device, nullptr, napi_enumerable),
                  InstanceAccessor("ptr", &MappedGLMemory::ptr, nullptr, napi_enumerable),
                  InstanceMethod("slice", &MappedGLMemory::slice),
                });
  MappedGLMemory::constructor = Napi::Persistent(ctor);
  MappedGLMemory::constructor.SuppressDestruct();

  exports.Set("MappedGLMemory", ctor);

  return exports;
}

MappedGLMemory::MappedGLMemory(CallbackArgs const& args)
  : Napi::ObjectWrap<MappedGLMemory>(args), Memory(args) {
  NODE_CUDA_EXPECT(args.IsConstructCall(), "MappedGLMemory constructor requires 'new'", args.Env());
  NODE_CUDA_EXPECT(args.Length() == 0 || (args.Length() == 1 && args[0].IsNumber()),
                   "MappedGLMemory constructor requires a numeric byteLength argument",
                   args.Env());
  if (args.Length() == 1) { Initialize(args[0]); }
}

Napi::Object MappedGLMemory::New(cudaGraphicsResource_t resource) {
  auto inst = MappedGLMemory::constructor.New({});
  MappedGLMemory::Unwrap(inst)->Initialize(resource);
  return inst;
}

void MappedGLMemory::Initialize(cudaGraphicsResource_t resource) {
  NODE_CUDA_TRY(cudaGraphicsResourceGetMappedPointer(&data_, &size_, resource), Env());
  Napi::MemoryManagement::AdjustExternalMemory(Env(), size_);
}

void MappedGLMemory::Finalize(Napi::Env env) {
  data_ = nullptr;
  size_ = 0;
}

Napi::Value MappedGLMemory::slice(Napi::CallbackInfo const& info) {
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
