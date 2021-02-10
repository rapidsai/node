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

#include <node_cudf/scalar.hpp>
#include <node_cudf/utilities/dtypes.hpp>

#include <nv_node/utilities/napi_to_cpp.hpp>

#include <cudf/types.hpp>

#include <napi.h>
#include <memory>
#include <type_traits>

namespace nv {

template <>
inline NapiToCPP::operator cudf::data_type() const {
  if (IsObject()) {
    auto obj = ToObject();
    if (obj.Has("typeId") && obj.Get("typeId").IsNumber()) {  //
      return arrow_to_cudf_type(obj);
    }
  }
  NAPI_THROW(Napi::Error::New(Env()), "Expected value to be a DataType");
}

template <>
inline NapiToCPP::operator cudf::duration_D() const {
  return cudf::duration_D{operator cudf::duration_D::rep()};
}

template <>
inline NapiToCPP::operator cudf::duration_s() const {
  return cudf::duration_s{operator cudf::duration_s::rep()};
}

template <>
inline NapiToCPP::operator cudf::duration_ms() const {
  return cudf::duration_ms{operator cudf::duration_ms::rep()};
}

template <>
inline NapiToCPP::operator cudf::duration_us() const {
  return cudf::duration_us{operator cudf::duration_us::rep()};
}

template <>
inline NapiToCPP::operator cudf::duration_ns() const {
  return cudf::duration_ns{operator cudf::duration_ns::rep()};
}

template <>
inline NapiToCPP::operator cudf::timestamp_D() const {
  return cudf::timestamp_D{operator cudf::duration_D()};
}

template <>
inline NapiToCPP::operator cudf::timestamp_s() const {
  return cudf::timestamp_s{operator cudf::duration_s()};
}

template <>
inline NapiToCPP::operator cudf::timestamp_ms() const {
  return cudf::timestamp_ms{operator cudf::duration_ms()};
}

template <>
inline NapiToCPP::operator cudf::timestamp_us() const {
  return cudf::timestamp_us{operator cudf::duration_us()};
}

template <>
inline NapiToCPP::operator cudf::timestamp_ns() const {
  return cudf::timestamp_ns{operator cudf::duration_ns()};
}

}  // namespace nv
