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

ConstructorReference DeviceBuffer::constructor;

Napi::Object DeviceBuffer::Init(Napi::Env env, Napi::Object exports) {
  exports.Set("DeviceBuffer", [&]() {
    (DeviceBuffer::constructor =
       Napi::Persistent(DefineClass(env,
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
                                    })))
      .SuppressDestruct();
    return DeviceBuffer::constructor.Value();
  }());
  return exports;
}

ObjectUnwrap<DeviceBuffer> DeviceBuffer::New(std::unique_ptr<rmm::device_buffer> buffer) {
  auto buf     = New(MemoryResource::Cuda(), buffer->stream());
  buf->buffer_ = std::move(buffer);
  return buf;
}

ObjectUnwrap<DeviceBuffer> DeviceBuffer::New(void* data,
                                             size_t size,
                                             ObjectUnwrap<MemoryResource> const& mr,
                                             rmm::cuda_stream_view stream) {
  NODE_CUDA_EXPECT(MemoryResource::is_instance(mr.object()),
                   "DeviceBuffer constructor requires a valid MemoryResource");
  return constructor.New(Span<char>(data, size), mr.object(), stream);
}

DeviceBuffer::DeviceBuffer(CallbackArgs const& args) : Napi::ObjectWrap<DeviceBuffer>(args) {
  auto& arg0 = args[0];
  auto& arg1 = args[1];
  auto& arg2 = args[2];
  auto input = arg0.IsObject()   ? arg0.operator Span<char>()
               : arg0.IsNumber() ? Span<char>(arg0.operator size_t())
                                 : Span<char>(0);

  if (MemoryResource::is_instance(arg1.val)) {
    mr_ = Napi::Persistent(arg1.ToObject());
  } else {
    mr_ = MemoryResource::Cuda().reference();
  }

  rmm::cuda_stream_view stream = arg2.IsNumber() ? arg2 : rmm::cuda_stream_default;

  switch (args.Length()) {
    case 0:
    case 1:
    case 2:
    case 3: {
      if (input.data() == nullptr || input.size() == 0) {
        buffer_.reset(new rmm::device_buffer(input.size(), stream, NapiToCPP(mr_.Value())));
      } else {
        buffer_.reset(new rmm::device_buffer(input, input.size(), stream, NapiToCPP(mr_.Value())));
        if (stream == rmm::cuda_stream_default) {
          NODE_CUDA_TRY(cudaStreamSynchronize(stream.value()), Env());
        }
      }
      Napi::MemoryManagement::AdjustExternalMemory(Env(), input.size());
      break;
    }
    default:
      NODE_CUDA_EXPECT(false,
                       "DeviceBuffer constructor requires a numeric size, and optional "
                       "stream and memory_resource arguments");
      break;
  }
}

void DeviceBuffer::Finalize(Napi::Env env) {
  if (size() > 0) { Napi::MemoryManagement::AdjustExternalMemory(env, -size()); }
}

ValueWrap<int32_t> DeviceBuffer::device() const {
  return MemoryResource::Unwrap(mr_.Value())->device();
}

Napi::Value DeviceBuffer::byte_length(Napi::CallbackInfo const& info) {
  return Napi::Number::New(info.Env(), buffer().size());
}

Napi::Value DeviceBuffer::capacity(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(buffer().capacity());
}

Napi::Value DeviceBuffer::is_empty(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(buffer().is_empty());
}

Napi::Value DeviceBuffer::ptr(Napi::CallbackInfo const& info) {
  return Napi::Number::New(info.Env(), reinterpret_cast<uintptr_t>(data()));
}

Napi::Value DeviceBuffer::resize(Napi::CallbackInfo const& info) {
  const CallbackArgs args{info};
  const size_t new_size = args[0];
  if (args.Length() > 1 && info[1].IsNumber()) {
    cudaStream_t stream = args[1];
    buffer().resize(new_size, stream);
  } else {
    buffer().resize(new_size);
  }
  return args.Env().Undefined();
}

Napi::Value DeviceBuffer::set_stream(Napi::CallbackInfo const& info) {
  const CallbackArgs args{info};
  const cudaStream_t stream = args[0];
  buffer().set_stream(stream);
  return args.Env().Undefined();
}

Napi::Value DeviceBuffer::shrink_to_fit(Napi::CallbackInfo const& info) {
  const CallbackArgs args{info};
  const cudaStream_t stream = args[0];
  buffer().shrink_to_fit(stream);
  return args.Env().Undefined();
}

Napi::Value DeviceBuffer::get_mr(Napi::CallbackInfo const& info) { return mr_.Value(); }

Napi::Value DeviceBuffer::slice(Napi::CallbackInfo const& info) {
  const CallbackArgs args{info};
  const size_t offset = args[0];
  size_t length       = size() - offset;
  if (args.Length() > 1 && info[1].IsNumber()) {
    length = args[1];
    length -= offset;
  }
  return DeviceBuffer::New(static_cast<char*>(data()) + offset, length, mr_.Value(), stream());
}

Napi::Value DeviceBuffer::device(Napi::CallbackInfo const& info) { return device(); }

Napi::Value DeviceBuffer::stream(Napi::CallbackInfo const& info) { return stream(); }

}  // namespace nv
