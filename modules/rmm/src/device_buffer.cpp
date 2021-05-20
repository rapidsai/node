// Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include "node_rmm/device_buffer.hpp"
#include "node_rmm/memory_resource.hpp"
#include "node_rmm/utilities/cpp_to_napi.hpp"
#include "node_rmm/utilities/napi_to_cpp.hpp"

#include <node_cuda/utilities/error.hpp>

namespace nv {

Napi::Function DeviceBuffer::Init(Napi::Env const& env, Napi::Object exports) {
  return DefineClass(env,
                     "DeviceBuffer",
                     {
                       InstanceAccessor<&DeviceBuffer::capacity>("capacity"),
                       InstanceAccessor<&DeviceBuffer::byte_length>("byteLength"),
                       InstanceAccessor<&DeviceBuffer::is_empty>("isEmpty"),
                       InstanceAccessor<&DeviceBuffer::ptr>("ptr"),
                       InstanceAccessor<&DeviceBuffer::device>("device"),
                       InstanceAccessor<&DeviceBuffer::stream>("stream"),
                       InstanceAccessor<&DeviceBuffer::get_mr>("memoryResource"),
                       InstanceMethod<&DeviceBuffer::resize>("resize"),
                       InstanceMethod<&DeviceBuffer::set_stream>("setStream"),
                       InstanceMethod<&DeviceBuffer::shrink_to_fit>("shrinkToFit"),
                       InstanceMethod<&DeviceBuffer::slice>("slice"),
                       InstanceMethod<&DeviceBuffer::dispose>("dispose"),
                     });
}

DeviceBuffer::wrapper_t DeviceBuffer::New(Napi::Env const& env,
                                          std::unique_ptr<rmm::device_buffer> buffer) {
  return New(env, std::move(buffer), MemoryResource::Current(env));
}

DeviceBuffer::wrapper_t DeviceBuffer::New(Napi::Env const& env,
                                          std::unique_ptr<rmm::device_buffer> buffer,
                                          MemoryResource::wrapper_t const& mr) {
  auto buf     = New(env, mr, buffer->stream());
  buf->buffer_ = std::move(buffer);
  return buf;
}

DeviceBuffer::wrapper_t DeviceBuffer::New(Napi::Env const& env,
                                          Napi::TypedArrayOf<uint8_t> const& data,
                                          MemoryResource::wrapper_t const& mr,
                                          rmm::cuda_stream_view stream) {
  NODE_CUDA_EXPECT(MemoryResource::IsInstance(mr),
                   "DeviceBuffer constructor requires a valid MemoryResource",
                   data.Env());
  return EnvLocalObjectWrap<DeviceBuffer>::New(env, data, mr, stream);
}

DeviceBuffer::wrapper_t DeviceBuffer::New(Napi::Env const& env,
                                          Span<char> const& data,
                                          MemoryResource::wrapper_t const& mr,
                                          rmm::cuda_stream_view stream) {
  NODE_CUDA_EXPECT(MemoryResource::IsInstance(mr),
                   "DeviceBuffer constructor requires a valid MemoryResource",
                   env);
  return EnvLocalObjectWrap<DeviceBuffer>::New(env, data, mr, stream);
}

DeviceBuffer::wrapper_t DeviceBuffer::New(Napi::Env const& env,
                                          void* const data,
                                          size_t const size,
                                          MemoryResource::wrapper_t const& mr,
                                          rmm::cuda_stream_view stream) {
  NODE_CUDA_EXPECT(MemoryResource::IsInstance(mr),
                   "DeviceBuffer constructor requires a valid MemoryResource",
                   env);
  return EnvLocalObjectWrap<DeviceBuffer>::New(env, Span<char>(data, size), mr, stream);
}

DeviceBuffer::DeviceBuffer(CallbackArgs const& args) : EnvLocalObjectWrap<DeviceBuffer>(args) {
  auto env   = args.Env();
  auto& arg0 = args[0];
  auto& arg1 = args[1];
  auto& arg2 = args[2];
  auto input = arg0.IsObject()   ? arg0.operator Span<char>()
               : arg0.IsNumber() ? Span<char>(arg0.operator size_t())
                                 : Span<char>(0);

  mr_ = Napi::Persistent<MemoryResource::wrapper_t>(
    MemoryResource::IsInstance(arg1) ? arg1.ToObject() : MemoryResource::Current(env));

  device_id_ = mr_.Value()->device();

  rmm::cuda_stream_view stream = arg2.IsNumber() ? arg2 : rmm::cuda_stream_default;

  switch (args.Length()) {
    case 0:
    case 1:
    case 2:
    case 3: {
      if (input.data() == nullptr || input.size() == 0) {
        Device::call_in_context(env, device().value(), [&] {
          buffer_ = std::make_unique<rmm::device_buffer>(input.size(), stream, get_mr());
        });
      } else {
        Device::call_in_context(env, device().value(), [&] {
          buffer_ = std::make_unique<rmm::device_buffer>(input, input.size(), stream, get_mr());
          if (stream == rmm::cuda_stream_default) {
            NODE_CUDA_TRY(cudaStreamSynchronize(stream.value()), env);
          }
        });
      }
      Napi::MemoryManagement::AdjustExternalMemory(env, size());
      break;
    }
    default:
      NODE_CUDA_EXPECT(false,
                       "DeviceBuffer constructor requires a numeric size, and optional "
                       "stream and memory_resource arguments",
                       env);
      break;
  }
}

void DeviceBuffer::Finalize(Napi::Env env) { dispose(env); }

void DeviceBuffer::dispose(Napi::Env env) {
  if (buffer_.get() != nullptr && capacity() > 0) {
    Napi::MemoryManagement::AdjustExternalMemory(env, -capacity());
    Device::call_in_context(env, device().value(), [&] { buffer_.reset(nullptr); });
  }
}

rmm::cuda_device_id DeviceBuffer::device() const noexcept { return device_id_; };

Napi::Value DeviceBuffer::byte_length(Napi::CallbackInfo const& info) {
  return Napi::Number::New(info.Env(), size());
}

Napi::Value DeviceBuffer::capacity(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), capacity());
}

Napi::Value DeviceBuffer::is_empty(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), is_empty());
}

Napi::Value DeviceBuffer::ptr(Napi::CallbackInfo const& info) {
  return Napi::Number::New(info.Env(), reinterpret_cast<uintptr_t>(data()));
}

void DeviceBuffer::resize(Napi::CallbackInfo const& info) {
  if (buffer_.get() != nullptr && info[0].IsNumber()) {
    auto const prev   = capacity();
    auto const size   = info[0].ToNumber().Int64Value();
    auto const stream = info[1].ToNumber().Int64Value();
    buffer_->resize(std::max(size, 0L), reinterpret_cast<cudaStream_t>(stream));
    Napi::MemoryManagement::AdjustExternalMemory(info.Env(), capacity() - prev);
  }
}

void DeviceBuffer::set_stream(Napi::CallbackInfo const& info) {
  if (buffer_.get() != nullptr) { buffer_->set_stream(CallbackArgs{info}[0]); }
}

void DeviceBuffer::shrink_to_fit(Napi::CallbackInfo const& info) {
  if (buffer_.get() != nullptr) { buffer_->shrink_to_fit(CallbackArgs{info}[0]); }
}

Napi::Value DeviceBuffer::get_mr(Napi::CallbackInfo const& info) { return mr_.Value(); }

Napi::Value DeviceBuffer::slice(Napi::CallbackInfo const& info) {
  CallbackArgs const args{info};
  size_t const offset = args[0];
  size_t length       = size() - offset;
  if (args.Length() > 1 && info[1].IsNumber()) {
    length = args[1];
    length -= offset;
  }
  return New(info.Env(), static_cast<char*>(data()) + offset, length, mr_.Value(), stream());
}

Napi::Value DeviceBuffer::device(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), device());
}

Napi::Value DeviceBuffer::stream(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), stream());
}

void DeviceBuffer::dispose(Napi::CallbackInfo const& info) { dispose(info.Env()); }

}  // namespace nv
