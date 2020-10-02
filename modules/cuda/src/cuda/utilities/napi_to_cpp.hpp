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

#include "cuda/device.hpp"
#include "napi.h"
#include "types.hpp"
#include "visit_struct/visit_struct.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nv_node/utilities/napi_to_cpp.hpp>

#include <type_traits>

namespace nv {

//
// CUDA Driver type conversion helpers
//
template <>
inline NapiToCPP::operator CUresult() const {
  return static_cast<CUresult>(this->operator int64_t());
}

template <>
inline NapiToCPP::operator CUfunction() const {
  return reinterpret_cast<CUfunction>(this->operator char*());
}

template <>
inline NapiToCPP::operator CUdevice_attribute() const {
  return static_cast<CUdevice_attribute>(this->operator int64_t());
}

template <>
inline NapiToCPP::operator CUpointer_attribute() const {
  return static_cast<CUpointer_attribute>(this->operator int64_t());
}

//
// CUDA Runtime type conversion helpers
//
template <>
inline NapiToCPP::operator cudaArray_t() const {
  return reinterpret_cast<cudaArray_t>(this->operator char*());
}

template <>
inline NapiToCPP::operator cudaGraphicsResource_t() const {
  return reinterpret_cast<cudaGraphicsResource_t>(this->operator char*());
}

template <>
inline NapiToCPP::operator cudaIpcMemHandle_t*() const {
  return reinterpret_cast<cudaIpcMemHandle_t*>(this->operator char*());
}

template <>
inline NapiToCPP::operator cudaStream_t() const {
  return reinterpret_cast<cudaStream_t>(this->operator char*());
}

template <>
inline NapiToCPP::operator cudaUUID_t*() const {
  return reinterpret_cast<cudaUUID_t*>(this->operator char*());
}

template <>
inline NapiToCPP::operator cudaDeviceProp() const {
  cudaDeviceProp props{};
  if (val.IsObject()) {
    auto obj = val.As<Napi::Object>();
    visit_struct::for_each(props, [&](char const* key, auto& val) {
      if (obj.Has(key) && !obj.Get(key).IsUndefined()) {
        using T                     = typename std::decay<decltype(val)>::type;
        *reinterpret_cast<T*>(&val) = NapiToCPP(obj.Get(key)).operator T();
      }
    });
  }
  return props;
}

template <>
inline NapiToCPP::operator Device() const {
  if (Device::is_instance(val)) { return *Device::Unwrap(val.ToObject()); }
  NAPI_THROW(Napi::Error::New(val.Env()), "Expected value to be a Device instance");
}

}  // namespace nv
