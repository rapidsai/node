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

#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <napi.h>

namespace nv {
namespace detail {

struct set_scalar_value {
  Napi::Value val;

  template <typename T>
  inline std::enable_if_t<cudf::is_numeric<T>(), void> operator()(
    std::unique_ptr<cudf::scalar>& scalar, cudaStream_t stream = 0) {
    static_cast<cudf::numeric_scalar<T>*>(scalar.get())->set_value(NapiToCPP(val), stream);
  }
  template <typename T>
  inline std::enable_if_t<std::is_same<T, cudf::string_view>::value, void> operator()(
    std::unique_ptr<cudf::scalar>& scalar, cudaStream_t stream = 0) {
    scalar.reset(new cudf::string_scalar(val.ToString(), true, stream));
  }
  template <typename T>
  inline std::enable_if_t<cudf::is_duration<T>(), void> operator()(
    std::unique_ptr<cudf::scalar>& scalar, cudaStream_t stream = 0) {
    static_cast<cudf::duration_scalar<T>*>(scalar.get())->set_value(NapiToCPP(val), stream);
  }
  template <typename T>
  inline std::enable_if_t<cudf::is_timestamp<T>(), void> operator()(
    std::unique_ptr<cudf::scalar>& scalar, cudaStream_t stream = 0) {
    static_cast<cudf::timestamp_scalar<T>*>(scalar.get())->set_value(NapiToCPP(val), stream);
  }
  template <typename T>
  inline std::enable_if_t<cudf::is_fixed_point<T>(), void> operator()(
    std::unique_ptr<cudf::scalar>& scalar, cudaStream_t stream = 0) {
    scalar.reset(new cudf::fixed_point_scalar<T>(val.ToNumber(), true, stream));
  }
  template <typename T>
  inline std::enable_if_t<std::is_same<T, cudf::list_view>::value, void> operator()(
    std::unique_ptr<cudf::scalar>& scalar, cudaStream_t stream = 0) {
    scalar.reset(new cudf::list_scalar(*Column::Unwrap(val.ToObject()), true, stream));
  }
  template <typename T>
  inline std::enable_if_t<!(cudf::is_numeric<T>() ||                      //
                            std::is_same<T, cudf::string_view>::value ||  //
                            cudf::is_duration<T>() ||                     //
                            cudf::is_timestamp<T>() ||                    //
                            cudf::is_fixed_point<T>() ||                  //
                            std::is_same<T, cudf::list_view>::value),
                          void>
  operator()(std::unique_ptr<cudf::scalar> const& scalar, cudaStream_t stream = 0) {
    NAPI_THROW(Napi::Error::New(val.Env(), "Unsupported dtype"));
  }
};

}  // namespace detail
}  // namespace nv
