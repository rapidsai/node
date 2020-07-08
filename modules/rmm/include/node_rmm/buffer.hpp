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

#include <rmm/device_buffer.hpp>

#include <napi.h>

namespace node_rmm {

class DeviceBuffer : public Napi::ObjectWrap<DeviceBuffer> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  static Napi::Value New(void* data, size_t size, cudaStream_t stream = 0);

  DeviceBuffer(Napi::CallbackInfo const& info);

  auto Stream() { return stream_; }
  auto& Buffer() { return buffer_; }
  size_t ByteLength() { return size_; }
  uint8_t* Data() { return static_cast<uint8_t*>(Buffer()->data()); }
  void Finalize(Napi::Env env) override;

 private:
  static Napi::FunctionReference constructor;

  Napi::Value GetByteLength(Napi::CallbackInfo const& info);
  Napi::Value GetPointer(Napi::CallbackInfo const& info);
  Napi::Value GetStream(Napi::CallbackInfo const& info);
  Napi::Value CopySlice(Napi::CallbackInfo const& info);

  std::unique_ptr<rmm::device_buffer> buffer_;
  size_t size_;
  cudaStream_t stream_;
};

}  // namespace node_rmm
