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

#include "array.hpp"
#include "cuda/utilities/cpp_to_napi.hpp"
#include "cuda/utilities/napi_to_cpp.hpp"
#include "macros.hpp"

#include <cuda_runtime_api.h>
#include <nv_node/utilities/args.hpp>

namespace nv {

Napi::FunctionReference CUDAArray::constructor;

Napi::Object CUDAArray::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor = DefineClass(
    env,
    "CUDAArray",
    {
      InstanceAccessor("channelFormatX", &CUDAArray::GetChannelFormatX, nullptr, napi_enumerable),
      InstanceAccessor("channelFormatY", &CUDAArray::GetChannelFormatY, nullptr, napi_enumerable),
      InstanceAccessor("channelFormatZ", &CUDAArray::GetChannelFormatZ, nullptr, napi_enumerable),
      InstanceAccessor("channelFormatW", &CUDAArray::GetChannelFormatW, nullptr, napi_enumerable),
      InstanceAccessor(
        "channelFormatKind", &CUDAArray::GetChannelFormatKind, nullptr, napi_enumerable),
      InstanceAccessor("width", &CUDAArray::GetWidth, nullptr, napi_enumerable),
      InstanceAccessor("height", &CUDAArray::GetHeight, nullptr, napi_enumerable),
      InstanceAccessor("depth", &CUDAArray::GetDepth, nullptr, napi_enumerable),
      InstanceAccessor("bytesPerElement", &CUDAArray::GetBytesPerElement, nullptr, napi_enumerable),
      InstanceAccessor("byteLength", &CUDAArray::GetByteLength, nullptr, napi_enumerable),
      InstanceAccessor("ary", &CUDAArray::GetPointer, nullptr, napi_enumerable),
    });
  CUDAArray::constructor = Napi::Persistent(ctor);
  CUDAArray::constructor.SuppressDestruct();
  return exports;
}

CUDAArray::CUDAArray(Napi::CallbackInfo const& info) : Napi::ObjectWrap<CUDAArray>(info) {}

Napi::Value CUDAArray::New(cudaArray_t array,
                           cudaExtent extent,
                           cudaChannelFormatDesc channelFormatDesc,
                           uint32_t flags,
                           array_type type) {
  auto ary                                   = CUDAArray::constructor.New({});
  CUDAArray::Unwrap(ary)->array_             = array;
  CUDAArray::Unwrap(ary)->extent_            = extent;
  CUDAArray::Unwrap(ary)->channelFormatDesc_ = channelFormatDesc;
  CUDAArray::Unwrap(ary)->flags_             = flags;
  CUDAArray::Unwrap(ary)->type_              = type;
  return ary;
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
