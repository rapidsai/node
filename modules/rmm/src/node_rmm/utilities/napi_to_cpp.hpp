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

#include <node_rmm/memory_resource.hpp>

#include <node_cuda/utilities/napi_to_cpp.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

namespace nv {

template <>
inline NapiToCPP::operator mr_type() const {
  return static_cast<mr_type>(this->operator uint8_t());
}

template <>
inline NapiToCPP::operator rmm::mr::device_memory_resource*() const {
  if (MemoryResource::IsInstance(val)) { return *MemoryResource::Unwrap(val.ToObject()); }
  if (val.IsNull() or val.IsUndefined()) { return rmm::mr::get_current_device_resource(); }
  NAPI_THROW(Napi::Error::New(val.Env()), "Expected value to be a MemoryResource instance");
}

template <>
inline NapiToCPP::operator rmm::cuda_device_id() const {
  if (this->IsNumber()) { return rmm::cuda_device_id{this->operator int32_t()}; }
  NAPI_THROW(Napi::Error::New(val.Env()), "Expected value to be a numeric device ordinal");
}

template <>
inline NapiToCPP::operator rmm::cuda_stream_view() const {
  if (this->IsNumber()) { return rmm::cuda_stream_view{this->operator cudaStream_t()}; }
  NAPI_THROW(Napi::Error::New(val.Env()), "Expected value to be a numeric cudaStream_t");
}

}  // namespace nv
