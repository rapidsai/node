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

#include <napi.h>

namespace node_nvencoder {

class ArrayInputFrame : public Napi::ObjectWrap<ArrayInputFrame> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  static Napi::Object New(NvEncInputFrame const* frame);
  ArrayInputFrame(Napi::CallbackInfo const& info);

 private:
  static Napi::FunctionReference constructor;
  NvEncInputFrame const& frame() { return *frame_; }
  Napi::Value GetArray(Napi::CallbackInfo const& info);
  Napi::Value GetPitch(Napi::CallbackInfo const& info);
  Napi::Value GetBufferFormat(Napi::CallbackInfo const& info);

  std::unique_ptr<NvEncInputFrame const> frame_{nullptr};
};

class BufferInputFrame : public Napi::ObjectWrap<BufferInputFrame> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  static Napi::Object New(NvEncInputFrame const* frame, size_t size);
  BufferInputFrame(Napi::CallbackInfo const& info);

 private:
  static Napi::FunctionReference constructor;
  NvEncInputFrame const& frame() { return *frame_; }

  Napi::Value GetBuffer(Napi::CallbackInfo const& info);
  Napi::Value GetByteLength(Napi::CallbackInfo const& info);
  Napi::Value GetPitch(Napi::CallbackInfo const& info);
  Napi::Value GetBufferFormat(Napi::CallbackInfo const& info);

  size_t size_{};
  std::unique_ptr<NvEncInputFrame const> frame_{nullptr};
};

class TextureInputFrame : public Napi::ObjectWrap<TextureInputFrame> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  static Napi::Object New(NvEncInputFrame const* frame);
  TextureInputFrame(Napi::CallbackInfo const& info);

 private:
  static Napi::FunctionReference constructor;
  NvEncInputFrame const& frame() { return *frame_; }

  Napi::Value GetTexture(Napi::CallbackInfo const& info);
  Napi::Value GetTarget(Napi::CallbackInfo const& info);
  Napi::Value GetPitch(Napi::CallbackInfo const& info);
  Napi::Value GetBufferFormat(Napi::CallbackInfo const& info);

  std::unique_ptr<NvEncInputFrame const> frame_{nullptr};
};

}  // namespace node_nvencoder
