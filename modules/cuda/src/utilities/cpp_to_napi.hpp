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

#include "types.hpp"
#include "visit_struct/visit_struct.hpp"

#include <cuda_runtime_api.h>
#include <nv_node/utilities/cpp_to_napi.hpp>

#include <string>
#include <type_traits>

namespace nv {

//
// CUDA Runtime type conversions
//

template <>
inline Napi::Value CPPToNapi::operator()(cudaUUID_t const& data) const {
  return this->operator()(data.bytes, sizeof(cudaUUID_t));
}

template <>
inline Napi::Value CPPToNapi::operator()(cudaError_t const& error) const {
  return Napi::Number::New(env, error);
}

template <>
inline Napi::Value CPPToNapi::operator()(cudaStream_t const& stream) const {
  return Napi::Number::New(env, reinterpret_cast<size_t>(stream));
}

template <>
inline Napi::Value CPPToNapi::operator()(cudaEvent_t const& event) const {
  return Napi::External<void>::New(env, event);
}

template <>
inline Napi::Value CPPToNapi::operator()(cudaGraph_t const& graph) const {
  return Napi::External<void>::New(env, graph);
}

template <>
inline Napi::Value CPPToNapi::operator()(cudaGraphNode_t const& graphNode) const {
  return Napi::External<void>::New(env, graphNode);
}

template <>
inline Napi::Value CPPToNapi::operator()(cudaGraphExec_t const& graphExec) const {
  return Napi::External<void>::New(env, graphExec);
}

template <>
inline Napi::Value CPPToNapi::operator()(cudaGraphicsResource_t const& resource) const {
  return Napi::External<void>::New(env, resource);
}

template <>
inline Napi::Value CPPToNapi::operator()(cudaIpcMemHandle_t const& data) const {
  return this->operator()(data.reserved, sizeof(cudaIpcMemHandle_t));
}

template <>
inline Napi::Value CPPToNapi::operator()(cudaDeviceProp const& props) const {
  auto cast_t = *this;
  auto obj    = Napi::Object::New(env);
  visit_struct::for_each(props, [&](char const* name, auto const& val) {  //
    using T = typename std::decay<decltype(val)>::type;
    if (std::is_pointer<T>()) {
      using P = typename std::remove_pointer<T>::type;
      if (std::is_same<P, char const>()) {
        obj.Set(name, cast_t(std::string{reinterpret_cast<char const*>(&val)}));
      } else {
        obj.Set(name, cast_t(reinterpret_cast<P*>(val), sizeof(val)));
      }
    } else {
      obj.Set(name, cast_t(val));
    }
  });
  return obj;
}

}  // namespace nv
