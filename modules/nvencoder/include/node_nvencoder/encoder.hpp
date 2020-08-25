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

#include "NvEncoder/NvEncoder.h"
#include "nvEncodeAPI.h"

#include <node_nvencoder/casting.hpp>

#include <napi.h>
#include <algorithm>
#include <cstdint>

namespace node_nvencoder {

class NvEncoderWrapper : public Napi::ObjectWrap<NvEncoderWrapper> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  static Napi::Value create(uint32_t width,
                            uint32_t height,
                            void* device,
                            NV_ENC_DEVICE_TYPE device_type);

  NvEncoderWrapper(Napi::CallbackInfo const& info);

  void Initialize(uint32_t width, uint32_t height, void* device, NV_ENC_DEVICE_TYPE device_type);
  void Finalize(Napi::Env env) override;

  NvEncoder& encoder() { return *encoder_; }

 private:
  static Napi::FunctionReference constructor;

  Napi::Value EncodeArray(Napi::CallbackInfo const& info);
  Napi::Value EncodeBuffer(Napi::CallbackInfo const& info);
  Napi::Value EncodeTexture(Napi::CallbackInfo const& info);

  std::unique_ptr<NvEncoder> encoder_{nullptr};
};

}  // namespace node_nvencoder
