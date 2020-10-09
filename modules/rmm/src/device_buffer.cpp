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

#include "device_buffer.hpp"
#include "cuda_memory_resource.hpp"
#include "macros.hpp"
#include "nv_node/utilities/args.hpp"
#include "nv_node/utilities/cpp_to_napi.hpp"
#include "rmm/mr/device/cuda_memory_resource.hpp"
#include "rmm/mr/device/device_memory_resource.hpp"
#include "rmm/mr/device/per_device_resource.hpp"

#include <node_cuda/utilities/napi_to_cpp.hpp>

namespace nv {

Napi::FunctionReference DeviceBuffer::constructor;

Napi::Object DeviceBuffer::Init(Napi::Env env, Napi::Object exports) {
  const Napi::Function ctor = DefineClass(
    env,
    "DeviceBuffer",
    {
      InstanceAccessor("capacity", &DeviceBuffer::capacity, nullptr, napi_enumerable),
      InstanceAccessor("byteLength", &DeviceBuffer::byteLength, nullptr, napi_enumerable),
      InstanceAccessor("isEmpty", &DeviceBuffer::isEmpty, nullptr, napi_enumerable),
      InstanceAccessor("ptr", &DeviceBuffer::ptr, nullptr, napi_enumerable),
      InstanceAccessor("stream", &DeviceBuffer::stream, nullptr, napi_enumerable),
      InstanceMethod("resize", &DeviceBuffer::resize),
      InstanceMethod("setStream", &DeviceBuffer::setStream),
      InstanceMethod("shrinkToFit", &DeviceBuffer::shrinkToFit),
      InstanceMethod("slice", &DeviceBuffer::slice),
    });
  DeviceBuffer::constructor = Napi::Persistent(ctor);
  DeviceBuffer::constructor.SuppressDestruct();
  exports.Set("DeviceBuffer", ctor);
  return exports;
}

Napi::Value DeviceBuffer::New(void* data,
                              size_t size,
                              cudaStream_t stream,
                              rmm::mr::cuda_memory_resource* mr) {
  const auto buf = DeviceBuffer::constructor.New({});
  DeviceBuffer::Unwrap(buf)->buffer_.reset(new rmm::device_buffer(data, size, stream));
  if (stream == NULL) { CUDA_TRY(buf.Env(), cudaStreamSynchronize(stream)); }
  Napi::MemoryManagement::AdjustExternalMemory(buf.Env(), size);
  return buf;
}

DeviceBuffer::DeviceBuffer(Napi::CallbackInfo const& info) : Napi::ObjectWrap<DeviceBuffer>(info) {
  const CallbackArgs args{info};
  const size_t size   = args[0];
  cudaStream_t stream = 0;
  if (args.Length() == 2 && info[1].IsNumber()) {
    stream = args[1];
    buffer_.reset(new rmm::device_buffer(size, stream));
  } else if (args.Length() >= 3 && info[1].IsNumber() && info[2].IsNumber()) {
    stream                 = args[1];
    CudaMemoryResource* mr = CudaMemoryResource::Unwrap(info[2].As<Napi::Object>());
    resource_              = mr->Resource();
    buffer_.reset(new rmm::device_buffer(size, stream, resource_.get()));
  }

  if (stream == NULL) { CUDA_TRY(info.Env(), cudaStreamSynchronize(stream)); }
}

void DeviceBuffer::Finalize(Napi::Env env) {
  const size_t size = Buffer()->size();
  if (buffer_.get() != nullptr && size > 0) {
    buffer_.reset(nullptr);
    Napi::MemoryManagement::AdjustExternalMemory(env, -size);
  }
  buffer_ = nullptr;
}

Napi::Value DeviceBuffer::byteLength(Napi::CallbackInfo const& info) {
  return Napi::Number::New(info.Env(), Buffer()->size());
}

Napi::Value DeviceBuffer::capacity(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(Buffer()->capacity());
}

Napi::Value DeviceBuffer::isEmpty(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(Buffer()->is_empty());
}

Napi::Value DeviceBuffer::ptr(Napi::CallbackInfo const& info) {
  return Napi::Number::New(info.Env(), reinterpret_cast<int64_t>(Data()));
}

Napi::Value DeviceBuffer::resize(Napi::CallbackInfo const& info) {
  const CallbackArgs args{info};
  const size_t new_size = args[0];
  if (args.Length() > 1 && info[1].IsNumber()) {
    cudaStream_t stream = args[1];
    Buffer()->resize(new_size, stream);
  } else {
    Buffer()->resize(new_size);
  }
  return args.Env().Undefined();
}

Napi::Value DeviceBuffer::setStream(Napi::CallbackInfo const& info) {
  const CallbackArgs args{info};
  const cudaStream_t stream = args[0];
  Buffer()->set_stream(stream);
  return args.Env().Undefined();
}

Napi::Value DeviceBuffer::shrinkToFit(Napi::CallbackInfo const& info) {
  const CallbackArgs args{info};
  const cudaStream_t stream = args[0];
  Buffer()->shrink_to_fit(stream);
  return args.Env().Undefined();
}

Napi::Value DeviceBuffer::slice(Napi::CallbackInfo const& info) {
  const CallbackArgs args{info};
  const size_t offset = args[0];
  size_t length       = Buffer()->size() - offset;
  if (args.Length() > 1 && info[1].IsNumber()) {
    length = args[1];
    length -= offset;
  }
  return DeviceBuffer::New(Data() + offset, length, Buffer()->stream());
}

Napi::Value DeviceBuffer::stream(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(Buffer()->stream());
}

}  // namespace nv
