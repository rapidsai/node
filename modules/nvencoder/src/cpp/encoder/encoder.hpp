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
#include "NvEncoder/NvEncoderCuda.h"
#include "NvEncoder/NvEncoderGL.h"

#include <napi.h>

#include <memory>
#include <vector>

namespace node_nvencoder {

class CUDANvEncoder : public Napi::ObjectWrap<CUDANvEncoder> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  static Napi::Value New(uint32_t encoderWidth,
                         uint32_t encoderHeight,
                         NV_ENC_BUFFER_FORMAT bufferFormat,
                         void* device = nullptr);

  CUDANvEncoder(Napi::CallbackInfo const& info);

  void Initialize(uint32_t encoderWidth,
                  uint32_t encoderHeight,
                  NV_ENC_BUFFER_FORMAT bufferFormat,
                  void* device = nullptr);

  NvEncoderCuda& encoder() { return *encoder_; }

 private:
  static Napi::FunctionReference constructor;

  Napi::Value EndEncode(Napi::CallbackInfo const& info);
  Napi::Value EncodeFrame(Napi::CallbackInfo const& info);
  Napi::Value CopyFromArray(Napi::CallbackInfo const& info);
  Napi::Value CopyFromHostBuffer(Napi::CallbackInfo const& info);
  Napi::Value CopyFromDeviceBuffer(Napi::CallbackInfo const& info);

  Napi::Value GetEncoderBufferCount(Napi::CallbackInfo const& info);

  CUcontext context_{nullptr};
  std::unique_ptr<NvEncoderCuda> encoder_{nullptr};
};

class GLNvEncoder : public Napi::ObjectWrap<GLNvEncoder> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  static Napi::Value New(uint32_t encoderWidth,
                         uint32_t encoderHeight,
                         NV_ENC_BUFFER_FORMAT bufferFormat);

  GLNvEncoder(Napi::CallbackInfo const& info);

  void Initialize(uint32_t encoderWidth, uint32_t encoderHeight, NV_ENC_BUFFER_FORMAT bufferFormat);

  NvEncoderGL& encoder() { return *encoder_; }

 private:
  static Napi::FunctionReference constructor;

  Napi::Value EndEncode(Napi::CallbackInfo const& info);
  Napi::Value EncodeFrame(Napi::CallbackInfo const& info);
  Napi::Value GetNextTextureInputFrame(Napi::CallbackInfo const& info);

  Napi::Value GetEncoderBufferCount(Napi::CallbackInfo const& info);

  std::unique_ptr<NvEncoderGL> encoder_{nullptr};
};

inline std::vector<napi_value> encoded_frame_array_buffers(
  Napi::Env env, std::vector<std::vector<uint8_t>> const& frames_) {
  std::vector<napi_value> results(frames_.size() + 1);
  results.at(0) = env.Null();
  std::transform(frames_.begin(), frames_.end(), results.begin() + 1, [&](auto& frame) {
    auto buf = Napi::ArrayBuffer::New(env, frame.size());
    auto ary = Napi::Uint8Array::New(env, buf.ByteLength(), buf, 0);
    std::memcpy(buf.Data(), frame.data(), frame.size());
    return ary;
  });
  return results;
}

class end_encode_worker : public Napi::AsyncWorker {
 public:
  end_encode_worker(NvEncoder* encoder, Napi::Function callback)
    : Napi::AsyncWorker(callback, "end_encode_worker"), encoder_{encoder} {}

 protected:
  void Execute() override { encoder_->EndEncode(frames_); }
  std::vector<napi_value> GetResult(Napi::Env env) override {
    return encoded_frame_array_buffers(env, frames_);
  }

 private:
  NvEncoder* encoder_;
  std::vector<std::vector<uint8_t>> frames_{};
};

class encode_frame_worker : public Napi::AsyncWorker {
 public:
  encode_frame_worker(NvEncoder* encoder, Napi::Function callback)
    : Napi::AsyncWorker(callback, "encode_frame_worker"), encoder_{encoder} {}

 protected:
  void Execute() override { encoder_->EncodeFrame(frames_); }
  std::vector<napi_value> GetResult(Napi::Env env) override {
    return encoded_frame_array_buffers(env, frames_);
  }

 private:
  NvEncoder* encoder_;
  std::vector<std::vector<uint8_t>> frames_{};
};

}  // namespace node_nvencoder
