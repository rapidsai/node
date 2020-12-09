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

#include <node_rmm/memory_resource.hpp>

#include <node_cuda/utilities/napi_to_cpp.hpp>

#include <nv_node/utilities/napi_to_cpp.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace nv {

template <>
inline NapiToCPP::operator rmm::mr::device_memory_resource*() const {
  if (CudaMemoryResource::is_instance(val)) {
    return CudaMemoryResource::Unwrap(val.ToObject())->get_mr().get();
  }
  if (ManagedMemoryResource::is_instance(val)) {
    return ManagedMemoryResource::Unwrap(val.ToObject())->get_mr().get();
  }
  if (PoolMemoryResource::is_instance(val)) {
    return PoolMemoryResource::Unwrap(val.ToObject())->get_mr().get();
  }
  if (FixedSizeMemoryResource::is_instance(val)) {
    return FixedSizeMemoryResource::Unwrap(val.ToObject())->get_mr().get();
  }
  if (BinningMemoryResource::is_instance(val)) {
    return BinningMemoryResource::Unwrap(val.ToObject())->get_mr().get();
  }
  if (LoggingResourceAdapter::is_instance(val)) {
    return LoggingResourceAdapter::Unwrap(val.ToObject())->get_mr().get();
  }
  NAPI_THROW(Napi::Error::New(val.Env()), "Expected value to be a MemoryResource instance");
}

template <>
inline NapiToCPP::operator rmm::cuda_stream_view() const {
  if (this->IsNumber()) { return rmm::cuda_stream_view{this->operator cudaStream_t()}; }
  NAPI_THROW(Napi::Error::New(val.Env()), "Expected value to be a numeric cudaStream_t");
}

}  // namespace nv
