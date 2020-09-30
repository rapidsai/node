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

#pragma once

#include <cuda_runtime_api.h>
#include <napi.h>

namespace nv {

enum class buffer_type : uint8_t { CUDA = 0, IPC = 1, GL = 2 };

class CUDABuffer : public Napi::ObjectWrap<CUDABuffer> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  static Napi::Value New(void* data, size_t size, buffer_type type = buffer_type::CUDA);

  CUDABuffer(Napi::CallbackInfo const& info);

  void* Data() { return data_; }
  size_t ByteLength() { return size_; }
  uint8_t* Begin() { return static_cast<uint8_t*>(data_); }
  void Finalize(Napi::Env env) override;

 private:
  static Napi::FunctionReference constructor;

  Napi::Value GetByteLength(Napi::CallbackInfo const& info);
  Napi::Value GetPointer(Napi::CallbackInfo const& info);
  Napi::Value CopySlice(Napi::CallbackInfo const& info);

  void* data_;
  size_t size_;
  buffer_type type_;
};

}  // namespace nv
