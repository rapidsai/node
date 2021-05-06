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

#pragma once

#include <nv_node/objectwrap.hpp>
#include <nv_node/utilities/args.hpp>

#include <cuda_runtime_api.h>

#include <napi.h>

namespace nv {

enum class array_type : uint8_t { CUDA = 0, IPC = 1, GL = 2 };

struct CUDAArray : public EnvLocalObjectWrap<CUDAArray> {
  /**
   * @brief Initialize and export the CUDAArray JavaScript constructor and prototype.
   *
   * @param env The active JavaScript environment.
   * @param exports The exports object to decorate.
   * @return Napi::Function The CUDAArray constructor function.
   */
  static Napi::Function Init(Napi::Env const& env, Napi::Object exports);

  /**
   * @brief Construct a new CUDAArray instance from C++.
   */
  static wrapper_t New(Napi::Env const& env,
                       cudaArray_t const& array,
                       cudaExtent const& extent,
                       cudaChannelFormatDesc const& channelFormatDesc,
                       uint32_t flags  = 0,
                       array_type type = array_type::CUDA);

  CUDAArray(CallbackArgs const& args);

  cudaArray_t Array() { return array_; }
  cudaExtent& Extent() { return extent_; }
  cudaChannelFormatDesc& ChannelFormatDesc() { return channelFormatDesc_; }
  uint32_t Flags() { return flags_; }
  uint32_t Width() { return std::max(Extent().width, size_t{1}); }
  uint32_t Height() { return std::max(Extent().height, size_t{1}); }
  uint32_t Depth() { return std::max(Extent().depth, size_t{1}); }
  uint8_t BytesPerElement() {
    auto x = ChannelFormatDesc().x;
    auto y = ChannelFormatDesc().y;
    auto z = ChannelFormatDesc().z;
    auto w = ChannelFormatDesc().w;
    return (x + y + z + w) >> 3;
  }

 private:
  Napi::Value GetPointer(Napi::CallbackInfo const& info);
  Napi::Value GetByteLength(Napi::CallbackInfo const& info);
  Napi::Value GetBytesPerElement(Napi::CallbackInfo const& info);
  // Napi::Value CopySlice(Napi::CallbackInfo const& info);

  Napi::Value GetWidth(Napi::CallbackInfo const& info);
  Napi::Value GetHeight(Napi::CallbackInfo const& info);
  Napi::Value GetDepth(Napi::CallbackInfo const& info);

  Napi::Value GetChannelFormatX(Napi::CallbackInfo const& info);
  Napi::Value GetChannelFormatY(Napi::CallbackInfo const& info);
  Napi::Value GetChannelFormatZ(Napi::CallbackInfo const& info);
  Napi::Value GetChannelFormatW(Napi::CallbackInfo const& info);
  Napi::Value GetChannelFormatKind(Napi::CallbackInfo const& info);

  cudaArray_t array_;
  uint32_t flags_;
  array_type type_;
  cudaExtent extent_;
  cudaChannelFormatDesc channelFormatDesc_;
};

}  // namespace nv
