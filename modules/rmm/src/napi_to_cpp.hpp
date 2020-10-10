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

#include "cuda_memory_resource.hpp"

#include <nv_node/utilities/napi_to_cpp.hpp>

namespace nv {

template <>
inline NapiToCPP::operator CudaMemoryResource() const {
  if (CudaMemoryResource::is_instance(val)) { return *CudaMemoryResource::Unwrap(val.ToObject()); }
  NAPI_THROW(Napi::Error::New(val.Env()), "Expected value to be a CudaMemoryResource instance");
}

template <>
inline NapiToCPP::operator rmm::mr::device_memory_resource*() const {
  if (CudaMemoryResource::is_instance(val)) {
    return this->operator CudaMemoryResource().Resource().get();
  }
  NAPI_THROW(Napi::Error::New(val.Env()), "Expected value to be a MemoryResource instance");
}

}  // namespace nv
