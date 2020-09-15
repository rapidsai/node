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

#include "addon.hpp"
#include "encoder/encoder.hpp"
#include "encoder/frame.hpp"
#include "macros.hpp"

#include <dlfcn.h>
#include <napi.h>
#include <nvEncodeAPI.h>
#include <nv_node/utilities/napi_to_cpp.hpp>

std::ostream& operator<<(std::ostream& os, const nv::NapiToCPP& self) {
  return os << self.operator std::string();
};

namespace nv {

Napi::Value init(Napi::CallbackInfo const& info) {
  if (!dlopen("libnvidia-encode.so.1", RTLD_LAZY)) {
    if (!dlopen("libnvidia-encode.so", RTLD_LAZY)) {
      throw Napi::Error::New(info.Env(), "libnvidia-encode.so not found");
    }
  }
  return info.This();
}

}  // namespace nv

Napi::Object init_module(Napi::Env env, Napi::Object exports) {
  EXPORT_FUNC(env, exports, "init", nv::init);
  nv::GLNvEncoder::Init(env, exports);
  nv::CUDANvEncoder::Init(env, exports);

  auto image = Napi::Object::New(env);
  EXPORT_PROP(exports, "image", image);
  nv::image::initModule(env, image);

  auto nvEncoderBufferFormats = Napi::Object::New(env);
  EXPORT_ENUM(env, nvEncoderBufferFormats, "UNDEFINED", NV_ENC_BUFFER_FORMAT_UNDEFINED);
  EXPORT_ENUM(env, nvEncoderBufferFormats, "NV12", NV_ENC_BUFFER_FORMAT_NV12);
  EXPORT_ENUM(env, nvEncoderBufferFormats, "YV12", NV_ENC_BUFFER_FORMAT_YV12);
  EXPORT_ENUM(env, nvEncoderBufferFormats, "IYUV", NV_ENC_BUFFER_FORMAT_IYUV);
  EXPORT_ENUM(env, nvEncoderBufferFormats, "YUV444", NV_ENC_BUFFER_FORMAT_YUV444);
  EXPORT_ENUM(env, nvEncoderBufferFormats, "YUV420_10BIT", NV_ENC_BUFFER_FORMAT_YUV420_10BIT);
  EXPORT_ENUM(env, nvEncoderBufferFormats, "YUV444_10BIT", NV_ENC_BUFFER_FORMAT_YUV444_10BIT);
  EXPORT_ENUM(env, nvEncoderBufferFormats, "ARGB", NV_ENC_BUFFER_FORMAT_ARGB);
  EXPORT_ENUM(env, nvEncoderBufferFormats, "ARGB10", NV_ENC_BUFFER_FORMAT_ARGB10);
  EXPORT_ENUM(env, nvEncoderBufferFormats, "AYUV", NV_ENC_BUFFER_FORMAT_AYUV);
  EXPORT_ENUM(env, nvEncoderBufferFormats, "ABGR", NV_ENC_BUFFER_FORMAT_ABGR);
  EXPORT_ENUM(env, nvEncoderBufferFormats, "ABGR10", NV_ENC_BUFFER_FORMAT_ABGR10);
  EXPORT_PROP(exports, "bufferFormats", nvEncoderBufferFormats);

  return exports;
}

NODE_API_MODULE(node_nvencoder, init_module);
