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

#include "encoder/encoder.hpp"
#include "macros.hpp"
#include "nvEncodeAPI.h"

#include <nv_node/utilities/args.hpp>
#include <nv_node/utilities/cpp_to_napi.hpp>

#include <napi.h>

namespace nv {

Napi::FunctionReference CUDANvEncoder::constructor;

Napi::Object CUDANvEncoder::Init(Napi::Env env, Napi::Object exports) {
  auto ctor = DefineClass(
    env,
    "CUDAEncoder",
    {InstanceMethod("close", &CUDANvEncoder::EndEncode, napi_enumerable),
     InstanceMethod("encode", &CUDANvEncoder::EncodeFrame, napi_enumerable),
     InstanceMethod("copyFromArray", &CUDANvEncoder::CopyFromArray, napi_enumerable),
     InstanceMethod("copyFromHostBuffer", &CUDANvEncoder::CopyFromHostBuffer, napi_enumerable),
     InstanceMethod("copyFromDeviceBuffer", &CUDANvEncoder::CopyFromDeviceBuffer, napi_enumerable),
     InstanceAccessor("frameSize", &CUDANvEncoder::GetFrameSize, nullptr, napi_enumerable),
     InstanceAccessor("bufferCount", &CUDANvEncoder::GetBufferCount, nullptr, napi_enumerable),
     InstanceAccessor("bufferFormat", &CUDANvEncoder::GetBufferFormat, nullptr, napi_enumerable)});

  CUDANvEncoder::constructor = Napi::Persistent(ctor);
  CUDANvEncoder::constructor.SuppressDestruct();

  exports.Set("CUDANvEncoder", ctor);

  return exports;
}

CUDANvEncoder::CUDANvEncoder(Napi::CallbackInfo const& info)
  : Napi::ObjectWrap<CUDANvEncoder>(info) {
  if (info.Length() != 1 || not info[0].IsObject()) {
    NAPI_THROW(Napi::Error::New(info.Env(), "CUDAEncoder: Invalid arguments, expected Object"));
  }
  auto opts = info[0].ToObject();
  if ((opts.Has("width") and opts.Get("width").IsNumber()) and
      (opts.Has("height") and opts.Get("height").IsNumber()) and
      (opts.Has("format") and opts.Get("format").IsNumber())) {
    size_t device   = NapiToCPP(opts.Get("device"));
    uint32_t format = NapiToCPP(opts.Get("format"));
    Initialize(NapiToCPP(opts.Get("width").ToNumber()),
               NapiToCPP(opts.Get("height").ToNumber()),
               static_cast<NV_ENC_BUFFER_FORMAT>(format),
               reinterpret_cast<void*>(device));
  } else {
    NAPI_THROW(Napi::Error::New(
      info.Env(), "CUDAEncoder: Invalid options, expected width, height, and format"));
  }
}

void CUDANvEncoder::Initialize(uint32_t encoderWidth,
                               uint32_t encoderHeight,
                               NV_ENC_BUFFER_FORMAT bufferFormat,
                               void* device) {
  // CUcontext context;
  if (device == nullptr) {
    CU_TRY(cuCtxGetCurrent(&context_));
  } else {
    context_ = reinterpret_cast<CUcontext>(device);
  }
  pixel_format_ = bufferFormat;
  encoder_.reset(new NvEncoderCuda(context_, encoderWidth, encoderHeight, bufferFormat));

  NV_ENC_INITIALIZE_PARAMS initializeParams = {NV_ENC_INITIALIZE_PARAMS_VER};
  NV_ENC_CONFIG encodeConfig                = {NV_ENC_CONFIG_VER};
  initializeParams.encodeWidth              = encoderWidth;
  initializeParams.encodeHeight             = encoderHeight;
  initializeParams.encodeConfig             = &encodeConfig;
  initializeParams.frameRateNum             = 60;
  initializeParams.frameRateDen             = 1;
  encoder_->CreateDefaultEncoderParams(&initializeParams,
                                       NV_ENC_CODEC_H264_GUID,         // TODO
                                       NV_ENC_PRESET_P3_GUID,          // TODO
                                       NV_ENC_TUNING_INFO_LOW_LATENCY  // TODO
  );

  encoder_->CreateEncoder(&initializeParams);
}

Napi::Value CUDANvEncoder::EndEncode(Napi::CallbackInfo const& info) {
  (new end_encode_worker(encoder_.get(), NapiToCPP(info[0])))->Queue();
  return info.Env().Undefined();
}

Napi::Value CUDANvEncoder::EncodeFrame(Napi::CallbackInfo const& info) {
  (new encode_frame_worker(encoder_.get(), NapiToCPP(info[0])))->Queue();
  return info.Env().Undefined();
}

Napi::Value CUDANvEncoder::CopyFromArray(Napi::CallbackInfo const& info) {
  // TODO
  // auto frame = encoder_->GetNextInputFrame();
  // encoder_->CopyToDeviceFrame(context_,
  //                             NapiToCPP(info[0]),
  //                             0,
  //                             reinterpret_cast<CUdeviceptr>(frame->inputPtr),
  //                             frame->pitch,
  //                             encoder_->GetEncodeWidth(),
  //                             encoder_->GetEncodeHeight(),
  //                             CU_MEMORYTYPE_DEVICE,
  //                             frame->bufferFormat,
  //                             frame->chromaOffsets,
  //                             frame->chromaPitch,
  //                             frame->numChromaPlanes);
  return info.Env().Undefined();
}

Napi::Value CUDANvEncoder::CopyFromHostBuffer(Napi::CallbackInfo const& info) {
  auto frame = encoder_->GetNextInputFrame();
  encoder_->CopyToDeviceFrame(context_,
                              NapiToCPP(info[0]),
                              0,
                              reinterpret_cast<CUdeviceptr>(frame->inputPtr),
                              frame->pitch,
                              encoder_->GetEncodeWidth(),
                              encoder_->GetEncodeHeight(),
                              CU_MEMORYTYPE_HOST,
                              frame->bufferFormat,
                              frame->chromaOffsets,
                              frame->chromaPitch,
                              frame->numChromaPlanes);
  return info.Env().Undefined();
}

Napi::Value CUDANvEncoder::CopyFromDeviceBuffer(Napi::CallbackInfo const& info) {
  auto frame = encoder_->GetNextInputFrame();
  encoder_->CopyToDeviceFrame(context_,
                              NapiToCPP(info[0]),
                              0,
                              reinterpret_cast<CUdeviceptr>(frame->inputPtr),
                              frame->pitch,
                              encoder_->GetEncodeWidth(),
                              encoder_->GetEncodeHeight(),
                              CU_MEMORYTYPE_DEVICE,
                              frame->bufferFormat,
                              frame->chromaOffsets,
                              frame->chromaPitch,
                              frame->numChromaPlanes);
  return info.Env().Undefined();
}

Napi::Value CUDANvEncoder::GetFrameSize(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(encoder_->GetFrameSize());
}

Napi::Value CUDANvEncoder::GetBufferCount(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(encoder_->GetEncoderBufferCount());
}

Napi::Value CUDANvEncoder::GetBufferFormat(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(pixel_format_);
}

}  // namespace nv
