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

#include <cudf/types.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>
#include <nv_node/utilities/cpp_to_napi.hpp>

#include <napi.h>

namespace nv {

template <>
inline Napi::Value CPPToNapi::operator()(cudf::type_id const& id) const {
  return Napi::Number::New(env, static_cast<int32_t>(id));
}

template <>
inline Napi::Value CPPToNapi::operator()(cudf::duration_D const& val) const {
  return (*this)(val.count());
}

template <>
inline Napi::Value CPPToNapi::operator()(cudf::duration_s const& val) const {
  return (*this)(val.count());
}

template <>
inline Napi::Value CPPToNapi::operator()(cudf::duration_ms const& val) const {
  return (*this)(val.count());
}

template <>
inline Napi::Value CPPToNapi::operator()(cudf::duration_us const& val) const {
  return (*this)(val.count());
}

template <>
inline Napi::Value CPPToNapi::operator()(cudf::duration_ns const& val) const {
  return (*this)(val.count());
}

template <>
inline Napi::Value CPPToNapi::operator()(cudf::timestamp_D const& val) const {
  return (*this)(val.time_since_epoch());
}

template <>
inline Napi::Value CPPToNapi::operator()(cudf::timestamp_s const& val) const {
  return (*this)(val.time_since_epoch());
}

template <>
inline Napi::Value CPPToNapi::operator()(cudf::timestamp_ms const& val) const {
  return (*this)(val.time_since_epoch());
}

template <>
inline Napi::Value CPPToNapi::operator()(cudf::timestamp_us const& val) const {
  return (*this)(val.time_since_epoch());
}

template <>
inline Napi::Value CPPToNapi::operator()(cudf::timestamp_ns const& val) const {
  return (*this)(val.time_since_epoch());
}

}  // namespace nv
