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

#include "buffer.hpp"
#include "macros.hpp"
#include "utilities/cpp_to_napi.hpp"
#include "utilities/napi_to_cpp.hpp"

#include <nv_node/utilities/args.hpp>

namespace nv {

Napi::FunctionReference CUDABuffer::constructor;

Napi::Object CUDABuffer::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor = DefineClass(
    env,
    "CUDABuffer",
    {
      InstanceAccessor("byteLength", &CUDABuffer::GetByteLength, nullptr, napi_enumerable),
      InstanceAccessor("ptr", &CUDABuffer::GetPointer, nullptr, napi_enumerable),
      InstanceMethod("slice", &CUDABuffer::CopySlice),
    });
  CUDABuffer::constructor = Napi::Persistent(ctor);
  CUDABuffer::constructor.SuppressDestruct();
  return exports;
}

Napi::Value CUDABuffer::New(void* data, size_t size, buffer_type type) {
  auto buf                       = CUDABuffer::constructor.New({});
  CUDABuffer::Unwrap(buf)->size_ = size;
  CUDABuffer::Unwrap(buf)->data_ = data;
  CUDABuffer::Unwrap(buf)->type_ = type;
  return buf;
}

CUDABuffer::CUDABuffer(Napi::CallbackInfo const& info) : Napi::ObjectWrap<CUDABuffer>(info) {}

void CUDABuffer::Finalize(Napi::Env env) {
  if (data_ != nullptr && size_ > 0) {
    switch (type_) {
      case buffer_type::GL: break;
      case buffer_type::IPC: CUDARTAPI::cudaIpcCloseMemHandle(data_); break;
      case buffer_type::CUDA:
        if (CUDARTAPI::cudaFree(data_) == cudaSuccess) {
          Napi::MemoryManagement::AdjustExternalMemory(env, -size_);
        }
        break;
    }
  }
  size_ = 0;
  data_ = nullptr;
}

Napi::Value CUDABuffer::GetByteLength(Napi::CallbackInfo const& info) {
  return Napi::Number::New(info.Env(), size_);
}

Napi::Value CUDABuffer::GetPointer(Napi::CallbackInfo const& info) {
  return Napi::Number::New(info.Env(), reinterpret_cast<int64_t>(Data()));
}

Napi::Value CUDABuffer::CopySlice(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  auto env      = info.Env();
  size_t offset = args[0];
  size_t length = size_ - offset;
  if (info.Length() > 1 && info[1].IsNumber()) {  //
    length = args[1].operator size_t() - offset;
  }
  void* data;
  CUDA_TRY(env, CUDARTAPI::cudaMalloc(&data, length));
  Napi::MemoryManagement::AdjustExternalMemory(env, length);
  CUDA_TRY(env, CUDARTAPI::cudaMemcpy(data, Begin() + offset, length, cudaMemcpyDefault));
  return CUDABuffer::New(data, length);
}

}  // namespace nv
