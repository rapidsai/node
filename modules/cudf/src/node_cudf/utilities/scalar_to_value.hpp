// Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <node_cudf/column.hpp>
#include <node_cudf/table.hpp>

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <napi.h>

namespace nv {
namespace detail {

struct get_scalar_value {
  Napi::Env env;

  template <typename T>
  inline std::enable_if_t<std::is_same_v<T, int8_t>, Napi::Value> operator()(
    std::unique_ptr<cudf::scalar> const& scalar, cudaStream_t stream = 0) {
    return scalar->is_valid(stream)
             ? Napi::Number::New(env,
                                 static_cast<cudf::numeric_scalar<T>*>(scalar.get())->value(stream))
             : env.Null();
  }
  template <typename T>
  inline std::enable_if_t<std::is_same_v<T, int16_t>, Napi::Value> operator()(
    std::unique_ptr<cudf::scalar> const& scalar, cudaStream_t stream = 0) {
    return scalar->is_valid(stream)
             ? Napi::Number::New(env,
                                 static_cast<cudf::numeric_scalar<T>*>(scalar.get())->value(stream))
             : env.Null();
  }
  template <typename T>
  inline std::enable_if_t<std::is_same_v<T, int32_t>, Napi::Value> operator()(
    std::unique_ptr<cudf::scalar> const& scalar, cudaStream_t stream = 0) {
    return scalar->is_valid(stream)
             ? Napi::Number::New(env,
                                 static_cast<cudf::numeric_scalar<T>*>(scalar.get())->value(stream))
             : env.Null();
  }
  template <typename T>
  inline std::enable_if_t<std::is_same_v<T, int64_t>, Napi::Value> operator()(
    std::unique_ptr<cudf::scalar> const& scalar, cudaStream_t stream = 0) {
    return scalar->is_valid(stream)
             ? Napi::BigInt::New(env,
                                 static_cast<cudf::numeric_scalar<T>*>(scalar.get())->value(stream))
             : env.Null();
  }

  template <typename T>
  inline std::enable_if_t<std::is_same_v<T, uint8_t>, Napi::Value> operator()(
    std::unique_ptr<cudf::scalar> const& scalar, cudaStream_t stream = 0) {
    return scalar->is_valid(stream)
             ? Napi::Number::New(env,
                                 static_cast<cudf::numeric_scalar<T>*>(scalar.get())->value(stream))
             : env.Null();
  }
  template <typename T>
  inline std::enable_if_t<std::is_same_v<T, uint16_t>, Napi::Value> operator()(
    std::unique_ptr<cudf::scalar> const& scalar, cudaStream_t stream = 0) {
    return scalar->is_valid(stream)
             ? Napi::Number::New(env,
                                 static_cast<cudf::numeric_scalar<T>*>(scalar.get())->value(stream))
             : env.Null();
  }
  template <typename T>
  inline std::enable_if_t<std::is_same_v<T, uint32_t>, Napi::Value> operator()(
    std::unique_ptr<cudf::scalar> const& scalar, cudaStream_t stream = 0) {
    return scalar->is_valid(stream)
             ? Napi::Number::New(env,
                                 static_cast<cudf::numeric_scalar<T>*>(scalar.get())->value(stream))
             : env.Null();
  }
  template <typename T>
  inline std::enable_if_t<std::is_same_v<T, uint64_t>, Napi::Value> operator()(
    std::unique_ptr<cudf::scalar> const& scalar, cudaStream_t stream = 0) {
    return scalar->is_valid(stream)
             ? Napi::BigInt::New(env,
                                 static_cast<cudf::numeric_scalar<T>*>(scalar.get())->value(stream))
             : env.Null();
  }
  template <typename T>
  inline std::enable_if_t<cudf::is_floating_point<T>(), Napi::Value> operator()(
    std::unique_ptr<cudf::scalar> const& scalar, cudaStream_t stream = 0) {
    return scalar->is_valid(stream)
             ? Napi::Number::New(env,
                                 static_cast<cudf::numeric_scalar<T>*>(scalar.get())->value(stream))
             : env.Null();
  }
  template <typename T>
  inline std::enable_if_t<std::is_same_v<T, bool>, Napi::Value> operator()(
    std::unique_ptr<cudf::scalar> const& scalar, cudaStream_t stream = 0) {
    return scalar->is_valid(stream)
             ? Napi::Boolean::New(
                 env, static_cast<cudf::numeric_scalar<bool>*>(scalar.get())->value(stream))
             : env.Null();
  }
  template <typename T>
  inline std::enable_if_t<std::is_same_v<T, cudf::string_view>, Napi::Value> operator()(
    std::unique_ptr<cudf::scalar> const& scalar, cudaStream_t stream = 0) {
    return scalar->is_valid(stream)
             ? Napi::Value::From(env,
                                 static_cast<cudf::string_scalar*>(scalar.get())->to_string(stream))
             : env.Null();
  }
  template <typename T>
  inline std::enable_if_t<cudf::is_duration<T>(), Napi::Value> operator()(
    std::unique_ptr<cudf::scalar> const& scalar, cudaStream_t stream = 0) {
    return scalar->is_valid(stream)
             ? Napi::Value::From(
                 env, static_cast<cudf::duration_scalar<T>*>(scalar.get())->value(stream))
             : env.Null();
  }
  template <typename T>
  inline std::enable_if_t<cudf::is_timestamp<T>(), Napi::Value> operator()(
    std::unique_ptr<cudf::scalar> const& scalar, cudaStream_t stream = 0) {
    return scalar->is_valid(stream)
             ? Napi::Value::From(
                 env, static_cast<cudf::timestamp_scalar<T>*>(scalar.get())->value(stream))
             : env.Null();
  }
  template <typename T>
  inline std::enable_if_t<std::is_same_v<T, numeric::decimal32>, Napi::Value> operator()(
    std::unique_ptr<cudf::scalar> const& scalar, cudaStream_t stream = 0) {
    return scalar->is_valid(stream)
             ? Napi::Value::From(
                 env, static_cast<cudf::fixed_point_scalar<T>*>(scalar.get())->value(stream))
             : env.Null();
  }
  template <typename T>
  inline std::enable_if_t<std::is_same_v<T, numeric::decimal64>, Napi::Value> operator()(
    std::unique_ptr<cudf::scalar> const& scalar, cudaStream_t stream = 0) {
    return scalar->is_valid(stream)
             ? Napi::Value::From(
                 env, static_cast<cudf::fixed_point_scalar<T>*>(scalar.get())->value(stream))
             : env.Null();
  }
  template <typename T>
  inline std::enable_if_t<std::is_same_v<T, cudf::list_view>, Napi::Value> operator()(
    std::unique_ptr<cudf::scalar> const& scalar, cudaStream_t stream = 0) {
    return scalar->is_valid(stream)
             ? Column::New(env,
                           std::make_unique<cudf::column>(
                             // The list_scalar's column_view is copied here because the underlying
                             // column cannot be moved.
                             static_cast<cudf::list_scalar*>(scalar.get())->view(),
                             stream))
             : env.Null();
  }
  template <typename T>
  inline std::enable_if_t<std::is_same_v<T, cudf::struct_view>, Napi::Value> operator()(
    std::unique_ptr<cudf::scalar> const& scalar, cudaStream_t stream = 0) {
    return scalar->is_valid(stream)
             ? Table::New(env,
                          std::make_unique<cudf::table>(
                            // The struct_scalar's table_view is copied here because the underlying
                            // table cannot be moved.
                            static_cast<cudf::struct_scalar*>(scalar.get())->view(),
                            stream))
             : env.Null();
  }
  template <typename T>
  inline std::enable_if_t<!(cudf::is_chrono<T>() ||                   //
                            cudf::is_index_type<T>() ||               //
                            cudf::is_floating_point<T>() ||           //
                            std::is_same_v<T, bool> ||                //
                            std::is_same_v<T, cudf::string_view> ||   //
                            std::is_same_v<T, numeric::decimal32> ||  //
                            std::is_same_v<T, numeric::decimal64> ||  //
                            std::is_same_v<T, cudf::list_view> ||     //
                            std::is_same_v<T, cudf::struct_view>),
                          Napi::Value>
  operator()(std::unique_ptr<cudf::scalar> const& scalar, cudaStream_t stream = 0) {
    NAPI_THROW(Napi::Error::New(env, "Unsupported dtype"));
  }
};

}  // namespace detail

}  // namespace nv

namespace Napi {

template <>
inline Value Value::From(napi_env env, std::unique_ptr<cudf::scalar> const& scalar) {
  return cudf::type_dispatcher(scalar->type(), nv::detail::get_scalar_value{env}, scalar);
}

}  // namespace Napi
