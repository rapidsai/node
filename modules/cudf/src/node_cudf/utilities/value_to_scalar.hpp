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
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <napi.h>

namespace nv {
namespace detail {

namespace scalar {
inline int64_t get_int64(Napi::Value const& val) {
  bool lossless = true;
  return !val.IsBigInt() ? val.ToNumber().Int64Value()
                         : val.As<Napi::BigInt>().Int64Value(&lossless);
}

inline uint64_t get_uint64(Napi::Value const& val) {
  bool lossless = true;
  return !val.IsBigInt() ? static_cast<uint64_t>(val.ToNumber().Int64Value())
                         : val.As<Napi::BigInt>().Uint64Value(&lossless);
}
}  // namespace scalar

struct set_scalar_value {
  Napi::Value val;

  template <typename T>
  inline std::enable_if_t<std::is_same_v<T, int8_t>, void> operator()(
    std::unique_ptr<cudf::scalar>& scalar, cudaStream_t stream = 0) {
    static_cast<cudf::numeric_scalar<T>*>(scalar.get())
      ->set_value(static_cast<T>(scalar::get_int64(val)), stream);
  }
  template <typename T>
  inline std::enable_if_t<std::is_same_v<T, int16_t>, void> operator()(
    std::unique_ptr<cudf::scalar>& scalar, cudaStream_t stream = 0) {
    static_cast<cudf::numeric_scalar<T>*>(scalar.get())
      ->set_value(static_cast<T>(scalar::get_int64(val)), stream);
  }
  template <typename T>
  inline std::enable_if_t<std::is_same_v<T, int32_t>, void> operator()(
    std::unique_ptr<cudf::scalar>& scalar, cudaStream_t stream = 0) {
    static_cast<cudf::numeric_scalar<T>*>(scalar.get())
      ->set_value(static_cast<T>(scalar::get_int64(val)), stream);
  }
  template <typename T>
  inline std::enable_if_t<std::is_same_v<T, int64_t>, void> operator()(
    std::unique_ptr<cudf::scalar>& scalar, cudaStream_t stream = 0) {
    static_cast<cudf::numeric_scalar<T>*>(scalar.get())
      ->set_value(static_cast<T>(scalar::get_int64(val)), stream);
  }
  template <typename T>
  inline std::enable_if_t<std::is_same_v<T, uint8_t>, void> operator()(
    std::unique_ptr<cudf::scalar>& scalar, cudaStream_t stream = 0) {
    static_cast<cudf::numeric_scalar<T>*>(scalar.get())
      ->set_value(static_cast<T>(scalar::get_uint64(val)), stream);
  }
  template <typename T>
  inline std::enable_if_t<std::is_same_v<T, uint16_t>, void> operator()(
    std::unique_ptr<cudf::scalar>& scalar, cudaStream_t stream = 0) {
    static_cast<cudf::numeric_scalar<T>*>(scalar.get())
      ->set_value(static_cast<T>(scalar::get_uint64(val)), stream);
  }
  template <typename T>
  inline std::enable_if_t<std::is_same_v<T, uint32_t>, void> operator()(
    std::unique_ptr<cudf::scalar>& scalar, cudaStream_t stream = 0) {
    static_cast<cudf::numeric_scalar<T>*>(scalar.get())
      ->set_value(static_cast<T>(scalar::get_uint64(val)), stream);
  }
  template <typename T>
  inline std::enable_if_t<std::is_same_v<T, uint64_t>, void> operator()(
    std::unique_ptr<cudf::scalar>& scalar, cudaStream_t stream = 0) {
    static_cast<cudf::numeric_scalar<T>*>(scalar.get())
      ->set_value(static_cast<T>(scalar::get_uint64(val)), stream);
  }
  template <typename T>
  inline std::enable_if_t<cudf::is_floating_point<T>(), void> operator()(
    std::unique_ptr<cudf::scalar>& scalar, cudaStream_t stream = 0) {
    static_cast<cudf::numeric_scalar<T>*>(scalar.get())->set_value(val.ToNumber(), stream);
  }
  template <typename T>
  inline std::enable_if_t<std::is_same_v<T, bool>, void> operator()(
    std::unique_ptr<cudf::scalar>& scalar, cudaStream_t stream = 0) {
    static_cast<cudf::numeric_scalar<T>*>(scalar.get())->set_value(val.ToBoolean(), stream);
  }
  template <typename T>
  inline std::enable_if_t<std::is_same_v<T, cudf::string_view>, void> operator()(
    std::unique_ptr<cudf::scalar>& scalar, cudaStream_t stream = 0) {
    scalar.reset(new cudf::string_scalar(val.ToString(), true, stream));
  }
  template <typename T>
  inline std::enable_if_t<cudf::is_duration<T>() and std::is_same_v<typename T::rep, int32_t>, void>
  operator()(std::unique_ptr<cudf::scalar>& scalar, cudaStream_t stream = 0) {
    static_cast<cudf::duration_scalar<T>*>(scalar.get())
      ->set_value(T{val.ToNumber().Int32Value()}, stream);
  }
  template <typename T>
  inline std::enable_if_t<cudf::is_duration<T>() and std::is_same_v<typename T::rep, int64_t>, void>
  operator()(std::unique_ptr<cudf::scalar>& scalar, cudaStream_t stream = 0) {
    static_cast<cudf::duration_scalar<T>*>(scalar.get())
      ->set_value(T{scalar::get_int64(val)}, stream);
  }
  template <typename T>
  inline std::enable_if_t<cudf::is_timestamp<T>() and std::is_same_v<typename T::rep, int32_t>,
                          void>
  operator()(std::unique_ptr<cudf::scalar>& scalar, cudaStream_t stream = 0) {
    static_cast<cudf::timestamp_scalar<T>*>(scalar.get())
      ->set_value(T{typename T::duration{val.ToNumber().Int32Value()}}, stream);
  }
  template <typename T>
  inline std::enable_if_t<cudf::is_timestamp<T>() and std::is_same_v<typename T::rep, int64_t>,
                          void>
  operator()(std::unique_ptr<cudf::scalar>& scalar, cudaStream_t stream = 0) {
    static_cast<cudf::timestamp_scalar<T>*>(scalar.get())
      ->set_value(T{typename T::duration{scalar::get_int64(val)}}, stream);
  }
  template <typename T>
  inline std::enable_if_t<std::is_same_v<T, numeric::decimal32>, void> operator()(
    std::unique_ptr<cudf::scalar>& scalar, cudaStream_t stream = 0) {
    scalar.reset(new cudf::fixed_point_scalar<T>(val.ToNumber(), true, stream));
  }
  template <typename T>
  inline std::enable_if_t<std::is_same_v<T, numeric::decimal64>, void> operator()(
    std::unique_ptr<cudf::scalar>& scalar, cudaStream_t stream = 0) {
    scalar.reset(new cudf::fixed_point_scalar<T>(val.ToNumber(), true, stream));
  }
  template <typename T>
  inline std::enable_if_t<std::is_same_v<T, cudf::list_view>, void> operator()(
    std::unique_ptr<cudf::scalar>& scalar, cudaStream_t stream = 0) {
    scalar.reset(new cudf::list_scalar(*Column::Unwrap(val.ToObject()), true, stream));
  }
  template <typename T>
  inline std::enable_if_t<std::is_same_v<T, cudf::struct_view>, void> operator()(
    std::unique_ptr<cudf::scalar>& scalar, cudaStream_t stream = 0) {
    scalar.reset(new cudf::struct_scalar(*Table::Unwrap(val.ToObject()), true, stream));
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
                          void>
  operator()(std::unique_ptr<cudf::scalar> const& scalar, cudaStream_t stream = 0) {
    NAPI_THROW(Napi::Error::New(val.Env(), "Unsupported dtype"));
  }
};

}  // namespace detail
}  // namespace nv
