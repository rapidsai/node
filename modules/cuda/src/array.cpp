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

#include <node_cuda/array.hpp>
#include <node_cuda/utilities/cpp_to_napi.hpp>
#include <node_cuda/utilities/napi_to_cpp.hpp>

#include <nv_node/macros.hpp>
#include <nv_node/utilities/args.hpp>

#include <cuda_runtime_api.h>

namespace nv {

Napi::Function CUDAArray::Init(Napi::Env const& env, Napi::Object exports) {
  auto ChannelFormatKind = Napi::Object::New(env);

  EXPORT_ENUM(env, ChannelFormatKind, "SIGNED", cudaChannelFormatKindSigned);
  EXPORT_ENUM(env, ChannelFormatKind, "UNSIGNED", cudaChannelFormatKindUnsigned);
  EXPORT_ENUM(env, ChannelFormatKind, "FLOAT", cudaChannelFormatKindFloat);
  EXPORT_ENUM(env, ChannelFormatKind, "NONE", cudaChannelFormatKindNone);
  EXPORT_ENUM(env, ChannelFormatKind, "lmemResizeToMax", cudaDeviceLmemResizeToMax);
  exports.Set("ChannelFormatKind", ChannelFormatKind);

  return DefineClass(env,
                     "CUDAArray",
                     {
                       InstanceAccessor<&CUDAArray::GetChannelFormatX>("channelFormatX"),
                       InstanceAccessor<&CUDAArray::GetChannelFormatY>("channelFormatY"),
                       InstanceAccessor<&CUDAArray::GetChannelFormatZ>("channelFormatZ"),
                       InstanceAccessor<&CUDAArray::GetChannelFormatW>("channelFormatW"),
                       InstanceAccessor<&CUDAArray::GetChannelFormatKind>("channelFormatKind"),
                       InstanceAccessor<&CUDAArray::GetWidth>("width"),
                       InstanceAccessor<&CUDAArray::GetHeight>("height"),
                       InstanceAccessor<&CUDAArray::GetDepth>("depth"),
                       InstanceAccessor<&CUDAArray::GetBytesPerElement>("bytesPerElement"),
                       InstanceAccessor<&CUDAArray::GetByteLength>("byteLength"),
                       InstanceAccessor<&CUDAArray::GetPointer>("ary"),
                     });
}

CUDAArray::CUDAArray(CallbackArgs const& args) : EnvLocalObjectWrap<CUDAArray>(args) {
  array_             = args[0];
  extent_            = args[1];
  channelFormatDesc_ = args[2];
  flags_             = args[3];
  type_              = args[4];
}

CUDAArray::wrapper_t CUDAArray::New(Napi::Env const& env,
                                    cudaArray_t const& array,
                                    cudaExtent const& extent,
                                    cudaChannelFormatDesc const& channelFormatDesc,
                                    uint32_t flags,
                                    array_type type) {
  return EnvLocalObjectWrap<CUDAArray>::New(
    env,
    {
      Napi::External<cudaArray_t>::New(env, const_cast<cudaArray_t*>(&array)),
      [&]() {
        auto obj = Napi::Object::New(env);
        obj.Set("width", extent.width);
        obj.Set("height", extent.height);
        obj.Set("depth", extent.depth);
        return obj;
      }(),
      [&]() {
        auto obj = Napi::Object::New(env);
        obj.Set("x", channelFormatDesc.x);
        obj.Set("y", channelFormatDesc.y);
        obj.Set("z", channelFormatDesc.z);
        obj.Set("w", channelFormatDesc.w);
        obj.Set("f", static_cast<uint8_t>(channelFormatDesc.f));
        return obj;
      }(),
      Napi::Number::New(env, flags),
      Napi::Number::New(env, static_cast<uint8_t>(type)),
    });
}

Napi::Value CUDAArray::GetBytesPerElement(Napi::CallbackInfo const& info) {
  return Napi::Number::New(info.Env(), BytesPerElement());
}

Napi::Value CUDAArray::GetByteLength(Napi::CallbackInfo const& info) {
  return Napi::Number::New(info.Env(), BytesPerElement() * Width() * Height() * Depth());
}

Napi::Value CUDAArray::GetPointer(Napi::CallbackInfo const& info) {
  return Napi::Number::New(info.Env(), reinterpret_cast<int64_t>(Array()));
}

Napi::Value CUDAArray::GetChannelFormatX(Napi::CallbackInfo const& info) {
  return Napi::Number::New(info.Env(), ChannelFormatDesc().x);
}

Napi::Value CUDAArray::GetChannelFormatY(Napi::CallbackInfo const& info) {
  return Napi::Number::New(info.Env(), ChannelFormatDesc().y);
}

Napi::Value CUDAArray::GetChannelFormatZ(Napi::CallbackInfo const& info) {
  return Napi::Number::New(info.Env(), ChannelFormatDesc().z);
}

Napi::Value CUDAArray::GetChannelFormatW(Napi::CallbackInfo const& info) {
  return Napi::Number::New(info.Env(), ChannelFormatDesc().w);
}

Napi::Value CUDAArray::GetChannelFormatKind(Napi::CallbackInfo const& info) {
  return Napi::Number::New(info.Env(), ChannelFormatDesc().f);
}

Napi::Value CUDAArray::GetWidth(Napi::CallbackInfo const& info) {
  return Napi::Number::New(info.Env(), Extent().width);
}

Napi::Value CUDAArray::GetHeight(Napi::CallbackInfo const& info) {
  return Napi::Number::New(info.Env(), Extent().height);
}

Napi::Value CUDAArray::GetDepth(Napi::CallbackInfo const& info) {
  return Napi::Number::New(info.Env(), Extent().depth);
}

}  // namespace nv
