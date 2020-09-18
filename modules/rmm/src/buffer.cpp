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

#include <nv_node/utilities/args.hpp>

namespace nv {

Napi::FunctionReference DeviceBuffer::constructor;

Napi::Object DeviceBuffer::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor = DefineClass(
    env,
    "DeviceBuffer",
    {
      InstanceAccessor("byteLength", &DeviceBuffer::GetByteLength, nullptr, napi_enumerable),
      InstanceAccessor("stream", &DeviceBuffer::GetStream, nullptr, napi_enumerable),
      InstanceAccessor("ptr", &DeviceBuffer::GetPointer, nullptr, napi_enumerable),
      InstanceMethod("slice", &DeviceBuffer::CopySlice),
    });
  DeviceBuffer::constructor = Napi::Persistent(ctor);
  DeviceBuffer::constructor.SuppressDestruct();
  exports.Set("DeviceBuffer", ctor);
  return exports;
}

Napi::Value DeviceBuffer::New(void* data, size_t size, cudaStream_t stream) {
  auto buf                         = DeviceBuffer::constructor.New({});
  DeviceBuffer::Unwrap(buf)->size_ = size;
  DeviceBuffer::Unwrap(buf)->buffer_.reset(new rmm::device_buffer(data, size, stream));
  if (stream == NULL) { CUDA_TRY(buf.Env(), cudaStreamSynchronize(stream)); }
  Napi::MemoryManagement::AdjustExternalMemory(buf.Env(), size);
  return buf;
}

DeviceBuffer::DeviceBuffer(Napi::CallbackInfo const& info) : Napi::ObjectWrap<DeviceBuffer>(info) {
  CallbackArgs args{info};
  this->size_   = 0;
  this->stream_ = 0;
  if (args.Length() >= 1 && info[0].IsNumber()) { this->size_ = args[0]; }
  if (args.Length() >= 2 && info[1].IsNumber()) { this->stream_ = args[1]; }
  this->buffer_.reset(new rmm::device_buffer(this->size_, this->stream_));
  if (this->stream_ == NULL) { CUDA_TRY(info.Env(), cudaStreamSynchronize(this->stream_)); }
}

void DeviceBuffer::Finalize(Napi::Env env) {
  if (buffer_.get() != nullptr && size_ > 0) {
    this->buffer_.reset(nullptr);
    Napi::MemoryManagement::AdjustExternalMemory(env, -size_);
  }
  size_   = 0;
  buffer_ = nullptr;
}

Napi::Value DeviceBuffer::GetByteLength(Napi::CallbackInfo const& info) {
  return Napi::Number::New(info.Env(), size_);
}

Napi::Value DeviceBuffer::GetPointer(Napi::CallbackInfo const& info) {
  return Napi::Number::New(info.Env(), reinterpret_cast<int64_t>(Data()));
}

Napi::Value DeviceBuffer::GetStream(Napi::CallbackInfo const& info) {
  return Napi::Number::New(info.Env(), reinterpret_cast<int64_t>(static_cast<void*>(stream_)));
}

Napi::Value DeviceBuffer::CopySlice(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  size_t offset = args[0];
  size_t length = size_ - offset;
  if (args.Length() > 1 && info[1].IsNumber()) {
    length = args[1];
    length -= offset;
  }
  return DeviceBuffer::New(Data() + offset, length, stream_);
}

}  // namespace nv
