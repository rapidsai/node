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

#include "NvEncoder/NvEncoder.h"
#include "NvEncoder/NvEncoderCuda.h"
#include "NvEncoder/NvEncoderGL.h"

#include <node_nvencoder/casting.hpp>
#include <node_nvencoder/encoder.hpp>
#include <node_nvencoder/macros.hpp>

#include <cuda.h>
#include <napi.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace node_nvencoder {

Napi::FunctionReference NvEncoderWrapper::constructor;

NvEncoderWrapper::NvEncoderWrapper(Napi::CallbackInfo const& info)
  : Napi::ObjectWrap<NvEncoderWrapper>(info) {
  if (info.Length() == 1 && info[0].IsObject()) {
    auto opts = info[0].ToObject();
    Initialize(FromJS(opts.Get("width")),
               FromJS(opts.Get("height")),
               reinterpret_cast<void*>(opts.Get("device").ToNumber().Int64Value()),
               static_cast<NV_ENC_DEVICE_TYPE>(opts.Get("deviceType").ToNumber().Uint32Value()));
  }
}

void NvEncoderWrapper::Initialize(uint32_t width,
                                  uint32_t height,
                                  void* device,
                                  NV_ENC_DEVICE_TYPE device_type) {
  if (device_type == NV_ENC_DEVICE_TYPE_OPENGL) {
    encoder_.reset(new NvEncoderGL(width, height, NV_ENC_BUFFER_FORMAT_ABGR));
  } else if (device_type == NV_ENC_DEVICE_TYPE_CUDA) {
    CUcontext context;
    if (device == nullptr) {
      CU_TRY(cuCtxGetCurrent(&context));
    } else {
      context = reinterpret_cast<CUcontext>(device);
    }
    encoder_.reset(new NvEncoderCuda(context, width, height, NV_ENC_BUFFER_FORMAT_ARGB));
  }

  NV_ENC_INITIALIZE_PARAMS initializeParams = {NV_ENC_INITIALIZE_PARAMS_VER};
  NV_ENC_CONFIG encodeConfig                = {NV_ENC_CONFIG_VER};
  initializeParams.encodeConfig             = &encodeConfig;
  encoder_->CreateDefaultEncoderParams(&initializeParams,
                                       NV_ENC_CODEC_H264_GUID,          // TODO
                                       NV_ENC_PRESET_P3_GUID,           // TODO
                                       NV_ENC_TUNING_INFO_HIGH_QUALITY  // TODO
  );

  encoder_->CreateEncoder(&initializeParams);
}

void NvEncoderWrapper::Finalize(Napi::Env env) { encoder_.reset(nullptr); }

Napi::Object NvEncoderWrapper::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor =
    DefineClass(env,
                "NvEncoder",
                {
                  InstanceMethod("encodeArray", &NvEncoderWrapper::EncodeArray),
                  InstanceMethod("encodeBuffer", &NvEncoderWrapper::EncodeBuffer),
                  InstanceMethod("encodeTexture", &NvEncoderWrapper::EncodeTexture),
                });

  NvEncoderWrapper::constructor = Napi::Persistent(ctor);
  NvEncoderWrapper::constructor.SuppressDestruct();

  exports.Set("NvEncoder", ctor);

  return exports;
}

Napi::Value NvEncoderWrapper::create(uint32_t width,
                                     uint32_t height,
                                     void* device,
                                     NV_ENC_DEVICE_TYPE device_type) {
  auto enc = NvEncoderWrapper::constructor.New({});

  NvEncoderWrapper::Unwrap(enc)->Initialize(width, height, device, device_type);

  return enc;
}

class encode_resource_worker : public Napi::AsyncWorker {
 public:
  encode_resource_worker(NvEncoderWrapper& wrapper,
                         Napi::Function const& callback,
                         bool encode_complete)
    : Napi::AsyncWorker(callback, "encode_frame_worker"),
      wrapper_{wrapper},
      end_encode_{encode_complete} {}

 protected:
  void Execute() override {
    if (end_encode_) {
      wrapper_.encoder().EndEncode(frames_);
    } else {
      wrapper_.encoder().EncodeFrame(frames_);
    }
  }

  std::vector<napi_value> GetResult(Napi::Env env) override {
    std::vector<napi_value> result(frames_.size() + 1);
    result.at(0) = env.Null();
    std::transform(frames_.begin(), frames_.end(), result.begin() + 1, [&](auto& frame) {
      auto buf = Napi::ArrayBuffer::New(env, frame.size());
      auto ary = Napi::Uint8Array::New(env, buf.ByteLength(), buf, 0);
      std::memcpy(buf.Data(), frame.data(), frame.size());
      return ary;
    });
    return result;
  }

  bool end_encode_;
  NvEncoderWrapper& wrapper_;
  std::vector<std::vector<uint8_t>> frames_;
};

Napi::Value NvEncoderWrapper::EncodeArray(Napi::CallbackInfo const& info) {
  // TODO
  return info.Env().Undefined();
}

Napi::Value NvEncoderWrapper::EncodeBuffer(Napi::CallbackInfo const& info) {
  encode_resource_worker* work;
  if (info[0].IsFunction()) {
    work = new encode_resource_worker(*this, FromJS(info[0]), true);
  } else {
    void* data;
    size_t size;
    std::tie(size, data)    = FromJS(info[0]).operator std::pair<size_t, uint8_t*>();
    Napi::Function callback = FromJS(info[1]);
    auto& frame             = const_cast<NvEncInputFrame&>(*encoder_->GetNextInputFrame());
    frame.bufferFormat      = NV_ENC_BUFFER_FORMAT_ARGB;
    frame.inputPtr          = data;
    frame.pitch             = encoder_->GetEncodeWidth() * 4;
    work                    = new encode_resource_worker(*this, callback, false);
  }
  work->Queue();
  return info.Env().Undefined();
}

Napi::Value NvEncoderWrapper::EncodeTexture(Napi::CallbackInfo const& info) {
  encode_resource_worker* work;
  if (info[0].IsFunction()) {
    work = new encode_resource_worker(*this, FromJS(info[0]), true);
  } else {
    uint32_t texture        = FromJS(info[0]);
    uint32_t target         = FromJS(info[1]);
    Napi::Function callback = FromJS(info[2]);
    auto& frame             = const_cast<NvEncInputFrame&>(*encoder_->GetNextInputFrame());
    auto texref             = reinterpret_cast<NV_ENC_INPUT_RESOURCE_OPENGL_TEX*>(frame.inputPtr);
    texref->target          = target;
    texref->texture         = texture;
    frame.bufferFormat      = NV_ENC_BUFFER_FORMAT_ABGR;
    frame.pitch             = encoder_->GetEncodeWidth() * 4;
    work                    = new encode_resource_worker(*this, callback, false);
  }
  work->Queue();
  return info.Env().Undefined();
}

}  // namespace node_nvencoder
