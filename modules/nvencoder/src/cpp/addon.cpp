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

#include <node_nvencoder/addon.hpp>
#include <node_nvencoder/encoder.hpp>
#include <node_nvencoder/macros.hpp>

#include <dlfcn.h>
#include <napi.h>
#include <nvEncodeAPI.h>

namespace node_nvencoder {

Napi::Value init(Napi::CallbackInfo const& info) {
  if (!dlopen("libnvidia-encode.so.1", RTLD_LAZY)) {
    if (!dlopen("libnvidia-encode.so", RTLD_LAZY)) {
      throw Napi::Error::New(info.Env(), "libnvidia-encode.so not found");
    }
  }
  return info.This();
}

}  // namespace node_nvencoder

Napi::Object init_module(Napi::Env env, Napi::Object exports) {
  EXPORT_FUNC(env, exports, "init", node_nvencoder::init);
  node_nvencoder::NvEncoderWrapper::Init(env, exports);
  return exports;
}

NODE_API_MODULE(node_nvencoder, init_module);
