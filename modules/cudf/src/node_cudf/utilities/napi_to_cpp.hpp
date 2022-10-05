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

#include <cudf/stream_compaction.hpp>
#include <cudf/strings/json.hpp>
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
inline NapiToCPP::operator cudf::sorted() const {
  if (IsBoolean()) { return ToBoolean() ? cudf::sorted::YES : cudf::sorted::NO; }
  NAPI_THROW(Napi::Error::New(Env()), "Expected value to be a boolean");
}

template <>
inline NapiToCPP::operator cudf::null_policy() const {
  if (IsBoolean()) { return ToBoolean() ? cudf::null_policy::INCLUDE : cudf::null_policy::EXCLUDE; }
  NAPI_THROW(Napi::Error::New(Env()), "Expected value to be a boolean");
}

template <>
inline NapiToCPP::operator cudf::null_order() const {
  if (IsNumber()) { return ToBoolean() ? cudf::null_order::BEFORE : cudf::null_order::AFTER; }
  NAPI_THROW(Napi::Error::New(Env()), "Expected value to be a NullOrder");
}

template <>
inline NapiToCPP::operator cudf::order() const {
  if (IsBoolean()) { return ToBoolean() ? cudf::order::ASCENDING : cudf::order::DESCENDING; }
  NAPI_THROW(Napi::Error::New(Env()), "Expected value to be a boolean");
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

template <>
inline NapiToCPP::operator cudf::interpolation() const {
  return static_cast<cudf::interpolation>(operator int32_t());
}

template <>
inline NapiToCPP::operator cudf::duplicate_keep_option() const {
  return static_cast<cudf::duplicate_keep_option>(operator int32_t());
}

template <>
inline NapiToCPP::operator cudf::strings::get_json_object_options() const {
  cudf::strings::get_json_object_options opts{};
  if (IsObject()) {
    auto obj = ToObject();
    opts.set_allow_single_quotes(obj.Get("allowSingleQuotes").ToBoolean());
    opts.set_missing_fields_as_nulls(obj.Get("missingFieldsAsNulls").ToBoolean());
    opts.set_strip_quotes_from_single_strings(obj.Get("stripQuotesFromSingleStrings").ToBoolean());
  }
  return opts;
}

}  // namespace nv
