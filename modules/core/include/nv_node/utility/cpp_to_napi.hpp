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

#include "nv_node/utility/span.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <napi.h>

#include <cstring>
#include <initializer_list>
#include <map>
#include <type_traits>
#include <vector>

namespace nv {

struct CPPToNapi {
  Napi::Env const env;
  inline CPPToNapi(Napi::Env const& env) : env(env) {}
  inline CPPToNapi(Napi::CallbackInfo const& info) : CPPToNapi(info.Env()) {}

  // Primitives
  inline Napi::Boolean operator()(bool const& val) const { return Napi::Boolean::New(env, val); }
  inline Napi::Number operator()(float const& val) const { return Napi::Number::New(env, val); }
  inline Napi::Number operator()(double const& val) const { return Napi::Number::New(env, val); }
  inline Napi::Number operator()(int8_t const& val) const { return Napi::Number::New(env, val); }
  inline Napi::Number operator()(int16_t const& val) const { return Napi::Number::New(env, val); }
  inline Napi::Number operator()(int32_t const& val) const { return Napi::Number::New(env, val); }
  inline Napi::Number operator()(int64_t const& val) const { return Napi::Number::New(env, val); }
  inline Napi::Number operator()(uint8_t const& val) const { return Napi::Number::New(env, val); }
  inline Napi::Number operator()(uint16_t const& val) const { return Napi::Number::New(env, val); }
  inline Napi::Number operator()(uint32_t const& val) const { return Napi::Number::New(env, val); }
  inline Napi::Number operator()(uint64_t const& val) const { return Napi::Number::New(env, val); }
  inline Napi::String operator()(std::string const& val) const {
    return Napi::String::New(env, val);
  }
  inline Napi::String operator()(std::u16string const& val) const {
    return Napi::String::New(env, val);
  }

  //
  // Arrays
  //
  template <typename T, int N>
  inline Napi::Array operator()(const T (&arr)[N]) const {
    return (*this)(std::vector<T>{arr, arr + N});
  }

  template <typename T>
  inline Napi::Array operator()(std::vector<T> const& vec) const {
    uint32_t idx = 0;
    auto arr     = Napi::Array::New(env, vec.size());
    std::for_each(
      vec.begin(), vec.end(), [&](T const& val) mutable { arr.Set((*this)(idx++), (*this)(val)); });
    return arr;
  }

  template <typename T>
  inline Napi::Array operator()(std::initializer_list<T> const& vec) const {
    uint32_t idx = 0;
    auto arr     = Napi::Array::New(env, vec.size());
    std::for_each(
      vec.begin(), vec.end(), [&](T const& val) mutable { arr.Set((*this)(idx++), (*this)(val)); });
    return arr;
  }

  //
  // Objects
  //
  template <typename T>
  inline Napi::Object operator()(std::vector<T> const& vals,
                                 std::vector<std::string> const& keys) const {
    auto self = *this;
    auto val  = vals.begin();
    auto key  = keys.begin();
    auto obj  = Napi::Object::New(env);
    while ((val != vals.end()) && (key != keys.end())) {
      obj.Set(self(*key), self(*val));
      std::advance(key, 1);
      std::advance(val, 1);
    }
    return obj;
  }

  // inline Napi::Object operator()(const CUDARTAPI::cudaDeviceProp& props) const {
  //   auto cast_t = *this;
  //   auto obj    = Napi::Object::New(env);
  //   visit_struct::for_each(
  //     props, [&](const char* name, const auto& value) { obj.Set(name, cast_t(value)); });
  //   return obj;
  // }

  //
  // Pointers
  //
  template <typename T>
  inline Napi::ArrayBuffer operator()(T const* data) const {
    return this->operator()(data, sizeof(T));
  }

  template <typename T>
  inline Napi::ArrayBuffer operator()(T const* data, size_t size) const {
    auto buf = Napi::ArrayBuffer::New(env, size);
    std::memcpy(buf.Data(), data, size);
    return buf;
  }

  template <typename T>
  inline Napi::Uint8Array operator()(Span<T> const& span) const {
    return Napi::ArrayBuffer::New(env, span.data(), span.size());
  }

  template <typename T, typename Finalizer>
  inline Napi::Uint8Array operator()(Span<T> const& span, Finalizer finalizer) const {
    return Napi::ArrayBuffer::New(env, span.data(), span.size(), finalizer);
  }

  //
  // CUDA Driver type conversions
  //
  // inline Napi::ArrayBuffer operator()(CUipcMemHandle const& data) const {
  //   return (*this)(data.reserved, sizeof(CUipcMemHandle));
  // }

  //
  // CUDA Runtime type conversions
  //
  inline Napi::ArrayBuffer operator()(cudaUUID_t const& data) const {
    return this->operator()(data.bytes, sizeof(cudaUUID_t));
  }

  inline Napi::Number operator()(cudaError_t const& error) const {
    return Napi::Number::New(env, error);
  }

  inline Napi::Number operator()(cudaStream_t const& stream) const {
    return Napi::Number::New(env, reinterpret_cast<size_t>(stream));
  }

  inline Napi::External<void> operator()(cudaEvent_t const& event) const {
    return Napi::External<void>::New(env, event);
  }

  inline Napi::External<void> operator()(cudaGraph_t const& graph) const {
    return Napi::External<void>::New(env, graph);
  }

  inline Napi::External<void> operator()(cudaGraphNode_t const& graphNode) const {
    return Napi::External<void>::New(env, graphNode);
  }

  inline Napi::External<void> operator()(cudaGraphExec_t const& graphExec) const {
    return Napi::External<void>::New(env, graphExec);
  }

  inline Napi::External<void> operator()(cudaFunction_t const& function) const {
    return Napi::External<void>::New(env, function);
  }

  inline Napi::External<void> operator()(cudaGraphicsResource_t const& resource) const {
    return Napi::External<void>::New(env, resource);
  }

  inline Napi::ArrayBuffer operator()(cudaIpcMemHandle_t const& data) const {
    return this->operator()(data.reserved, sizeof(cudaIpcMemHandle_t));
  }
};

}  // namespace nv
