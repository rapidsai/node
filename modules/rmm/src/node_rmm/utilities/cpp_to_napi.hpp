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

#include "../types.hpp"

#include <node_cuda/utilities/cpp_to_napi.hpp>

#include <rmm/mr/device/per_device_resource.hpp>

namespace nv {

template <>
inline Napi::Value CPPToNapi::operator()(mr_type const& type) const {
  return this->operator()(static_cast<uint8_t>(type));
}

template <>
inline Napi::Value CPPToNapi::operator()(rmm::cuda_device_id const& device) const {
  return this->operator()(device.value());
}

template <>
inline Napi::Value CPPToNapi::operator()(rmm::cuda_stream_view const& stream) const {
  return this->operator()(stream.value());
}

}  // namespace nv

namespace Napi {

template <>
inline Value Value::From(napi_env env, nv::mr_type const& type) {
  return Value::From(env, static_cast<uint8_t>(type));
}

template <>
inline Value Value::From(napi_env env, rmm::cuda_device_id const& device) {
  return Value::From(env, static_cast<int32_t>(device.value()));
}

template <>
inline Value Value::From(napi_env env, rmm::cuda_stream_view const& stream) {
  return Value::From(env, reinterpret_cast<uintptr_t>(stream.value()));
}

}  // namespace Napi
