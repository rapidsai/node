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

#include "scalar_to_value.hpp"

#include <nv_node/utilities/cpp_to_napi.hpp>

#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <napi.h>
#include <cuda/std/chrono>

namespace nv {

template <>
inline Napi::Value CPPToNapi::operator()(cudf::type_id const& id) const {
  return Napi::Number::New(Env(), static_cast<int32_t>(id));
}

template <>
inline Napi::Value CPPToNapi::operator()(cudf::duration_D const& val) const {
  return (*this)(val.count());
}

template <>
inline Napi::Value CPPToNapi::operator()(cudf::duration_s const& val) const {
  return Napi::BigInt::New(Env(), val.count());
}

template <>
inline Napi::Value CPPToNapi::operator()(cudf::duration_ms const& val) const {
  return Napi::BigInt::New(Env(), val.count());
}

template <>
inline Napi::Value CPPToNapi::operator()(cudf::duration_us const& val) const {
  return Napi::BigInt::New(Env(), val.count());
}

template <>
inline Napi::Value CPPToNapi::operator()(cudf::duration_ns const& val) const {
  return Napi::BigInt::New(Env(), val.count());
}

template <>
inline Napi::Value CPPToNapi::operator()(cudf::timestamp_D const& val) const {
  return Napi::Number::New(
    Env(), cuda::std::chrono::duration_cast<cudf::duration_ms>(val.time_since_epoch()).count());
}

template <>
inline Napi::Value CPPToNapi::operator()(cudf::timestamp_s const& val) const {
  return Napi::BigInt::New(
    Env(), cuda::std::chrono::duration_cast<cudf::duration_ms>(val.time_since_epoch()).count());
}

template <>
inline Napi::Value CPPToNapi::operator()(cudf::timestamp_ms const& val) const {
  return Napi::BigInt::New(
    Env(), cuda::std::chrono::duration_cast<cudf::duration_ms>(val.time_since_epoch()).count());
}

template <>
inline Napi::Value CPPToNapi::operator()(cudf::timestamp_us const& val) const {
  return Napi::BigInt::New(
    Env(), cuda::std::chrono::duration_cast<cudf::duration_ms>(val.time_since_epoch()).count());
}

template <>
inline Napi::Value CPPToNapi::operator()(cudf::timestamp_ns const& val) const {
  return Napi::BigInt::New(
    Env(), cuda::std::chrono::duration_cast<cudf::duration_ms>(val.time_since_epoch()).count());
}

template <>
inline Napi::Value CPPToNapi::operator()(std::unique_ptr<cudf::scalar> const& scalar) const {
  return cudf::type_dispatcher(scalar->type(), detail::get_scalar_value{Env()}, scalar);
}

}  // namespace nv

namespace Napi {

template <>
inline Value Value::From(napi_env env, cudf::type_id const& id) {
  return Value::From(env, static_cast<int32_t>(id));
}

template <>
inline Value Value::From(napi_env env, cudf::duration_D const& val) {
  return Value::From(env, val.count());
}

template <>
inline Value Value::From(napi_env env, cudf::duration_s const& val) {
  return Napi::BigInt::New(env, val.count());
}

template <>
inline Value Value::From(napi_env env, cudf::duration_ms const& val) {
  return Napi::BigInt::New(env, val.count());
}

template <>
inline Value Value::From(napi_env env, cudf::duration_us const& val) {
  return Napi::BigInt::New(env, val.count());
}

template <>
inline Value Value::From(napi_env env, cudf::duration_ns const& val) {
  return Napi::BigInt::New(env, val.count());
}

template <>
inline Value Value::From(napi_env env, cudf::timestamp_D const& val) {
  return Napi::Number::New(
    env, cuda::std::chrono::duration_cast<cudf::duration_ms>(val.time_since_epoch()).count());
}

template <>
inline Value Value::From(napi_env env, cudf::timestamp_s const& val) {
  return Napi::BigInt::New(
    env, cuda::std::chrono::duration_cast<cudf::duration_ms>(val.time_since_epoch()).count());
}

template <>
inline Value Value::From(napi_env env, cudf::timestamp_ms const& val) {
  return Napi::BigInt::New(
    env, cuda::std::chrono::duration_cast<cudf::duration_ms>(val.time_since_epoch()).count());
}

template <>
inline Value Value::From(napi_env env, cudf::timestamp_us const& val) {
  return Napi::BigInt::New(
    env, cuda::std::chrono::duration_cast<cudf::duration_ms>(val.time_since_epoch()).count());
}

template <>
inline Value Value::From(napi_env env, cudf::timestamp_ns const& val) {
  return Napi::BigInt::New(
    env, cuda::std::chrono::duration_cast<cudf::duration_ms>(val.time_since_epoch()).count());
}

}  // namespace Napi
