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
#include "encoder/frame.hpp"

#include <node_nvencoder/casting.hpp>
#include <node_nvencoder/macros.hpp>

#include <napi.h>

namespace node_nvencoder {

Napi::FunctionReference GLNvEncoder::constructor;

Napi::Object GLNvEncoder::Init(Napi::Env env, Napi::Object exports) {
  auto ctor = DefineClass(
    env,
    "GLEncoder",
    {InstanceMethod("close", &GLNvEncoder::EndEncode, napi_enumerable),
     InstanceMethod("encode", &GLNvEncoder::EncodeFrame, napi_enumerable),
     InstanceMethod("texture", &GLNvEncoder::GetNextTextureInputFrame, napi_enumerable),
     InstanceAccessor(
       "encoderBufferCount", &GLNvEncoder::GetEncoderBufferCount, nullptr, napi_enumerable)});

  GLNvEncoder::constructor = Napi::Persistent(ctor);
  GLNvEncoder::constructor.SuppressDestruct();

  exports.Set("GLNvEncoder", ctor);

  TextureInputFrame::Init(env, exports);

  return exports;
}

GLNvEncoder::GLNvEncoder(Napi::CallbackInfo const& info) : Napi::ObjectWrap<GLNvEncoder>(info) {
  if (info.Length() != 1 || not info[0].IsObject()) {
    NAPI_THROW(Napi::Error::New(info.Env(), "GLEncoder: Invalid arguments, expected Object"));
  }
  auto opts = info[0].ToObject();
  if ((opts.Has("width") and opts.Get("width").IsNumber()) and
      (opts.Has("height") and opts.Get("height").IsNumber()) and
      (opts.Has("format") and opts.Get("format").IsNumber())) {
    Initialize(FromJS(opts.Get("width").ToNumber()),
               FromJS(opts.Get("height").ToNumber()),
               FromJS(opts.Get("format").ToNumber()));
  } else {
    NAPI_THROW(Napi::Error::New(info.Env(),
                                "GLEncoder: Invalid options, expected width, height, and format"));
  }
}

void GLNvEncoder::Initialize(uint32_t encoderWidth,
                             uint32_t encoderHeight,
                             NV_ENC_BUFFER_FORMAT bufferFormat) {
  encoder_.reset(new NvEncoderGL(encoderWidth, encoderHeight, bufferFormat));

  NV_ENC_INITIALIZE_PARAMS initializeParams = {NV_ENC_INITIALIZE_PARAMS_VER};
  NV_ENC_CONFIG encodeConfig                = {NV_ENC_CONFIG_VER};
  initializeParams.encodeWidth              = encoderWidth;
  initializeParams.encodeHeight             = encoderHeight;
  initializeParams.encodeConfig             = &encodeConfig;
  encoder_->CreateDefaultEncoderParams(&initializeParams,
                                       NV_ENC_CODEC_H264_GUID,          // TODO
                                       NV_ENC_PRESET_P3_GUID,           // TODO
                                       NV_ENC_TUNING_INFO_HIGH_QUALITY  // TODO
  );

  encoder_->CreateEncoder(&initializeParams);
}

Napi::Value GLNvEncoder::EndEncode(Napi::CallbackInfo const& info) {
  (new end_encode_worker(encoder_.get(), FromJS(info[0])))->Queue();
  return info.Env().Undefined();
}

Napi::Value GLNvEncoder::EncodeFrame(Napi::CallbackInfo const& info) {
  (new encode_frame_worker(encoder_.get(), FromJS(info[0])))->Queue();
  return info.Env().Undefined();
}

Napi::Value GLNvEncoder::GetNextTextureInputFrame(Napi::CallbackInfo const& info) {
  return TextureInputFrame::New(encoder_->GetNextInputFrame());
}

Napi::Value GLNvEncoder::GetEncoderBufferCount(Napi::CallbackInfo const& info) {
  return ToNapi(info.Env())(encoder_->GetEncoderBufferCount());
}

/**
 *
 * TextureInputFrame
 *
 */

Napi::FunctionReference TextureInputFrame::constructor;

Napi::Object TextureInputFrame::Init(Napi::Env env, Napi::Object exports) {
  auto ctor = DefineClass(
    env,
    "TextureInputFrame",
    {InstanceAccessor("texture", &TextureInputFrame::GetTexture, nullptr, napi_enumerable),
     InstanceAccessor("target", &TextureInputFrame::GetTarget, nullptr, napi_enumerable),
     InstanceAccessor("pitch", &TextureInputFrame::GetPitch, nullptr, napi_enumerable),
     InstanceAccessor(
       "bufferFormat", &TextureInputFrame::GetBufferFormat, nullptr, napi_enumerable)});

  TextureInputFrame::constructor = Napi::Persistent(ctor);
  TextureInputFrame::constructor.SuppressDestruct();

  return exports;
}

TextureInputFrame::TextureInputFrame(Napi::CallbackInfo const& info)
  : Napi::ObjectWrap<TextureInputFrame>(info) {}

Napi::Object TextureInputFrame::New(NvEncInputFrame const* frame) {
  auto tex = TextureInputFrame::constructor.New({});
  TextureInputFrame::Unwrap(tex)->frame_.reset(frame);
  return tex;
}

Napi::Value TextureInputFrame::GetTexture(Napi::CallbackInfo const& info) {
  return ToNapi(info.Env())(
    frame_ == nullptr ? 0
                      : static_cast<NV_ENC_INPUT_RESOURCE_OPENGL_TEX*>(frame_->inputPtr)->texture);
}

Napi::Value TextureInputFrame::GetTarget(Napi::CallbackInfo const& info) {
  return ToNapi(info.Env())(
    frame_ == nullptr ? 0
                      : static_cast<NV_ENC_INPUT_RESOURCE_OPENGL_TEX*>(frame_->inputPtr)->target);
}

Napi::Value TextureInputFrame::GetPitch(Napi::CallbackInfo const& info) {
  return ToNapi(info.Env())(frame_ == nullptr ? 0 : frame_->pitch);
}

Napi::Value TextureInputFrame::GetBufferFormat(Napi::CallbackInfo const& info) {
  return ToNapi(info.Env())(frame_ == nullptr ? 0 : frame_->bufferFormat);
}

}  // namespace node_nvencoder
