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

#include <node_cudf/column.hpp>
#include <node_cudf/scalar.hpp>

#include <cudf/filling.hpp>
#include <cudf/replace.hpp>

namespace nv {

Column::wrapper_t Column::replace_nulls(cudf::column_view const& replacement,
                                        rmm::mr::device_memory_resource* mr) {
  try {
    return Column::New(Env(), cudf::replace_nulls(*this, replacement, mr));
  } catch (std::exception const& e) { NAPI_THROW(Napi::Error::New(Env(), e.what())); }
}

Column::wrapper_t Column::replace_nulls(cudf::scalar const& replacement,
                                        rmm::mr::device_memory_resource* mr) {
  try {
    return Column::New(Env(), cudf::replace_nulls(*this, replacement, mr));
  } catch (std::exception const& e) { NAPI_THROW(Napi::Error::New(Env(), e.what())); }
}

Column::wrapper_t Column::replace_nulls(cudf::replace_policy const& replace_policy,
                                        rmm::mr::device_memory_resource* mr) {
  try {
    return Column::New(Env(), cudf::replace_nulls(*this, replace_policy, mr));
  } catch (std::exception const& e) { NAPI_THROW(Napi::Error::New(Env(), e.what())); }
}

Column::wrapper_t Column::replace_nans(cudf::column_view const& replacement,
                                       rmm::mr::device_memory_resource* mr) {
  try {
    return Column::New(Env(), cudf::replace_nans(*this, replacement, mr));
  } catch (std::exception const& e) { NAPI_THROW(Napi::Error::New(Env(), e.what())); }
}

Column::wrapper_t Column::replace_nans(cudf::scalar const& replacement,
                                       rmm::mr::device_memory_resource* mr) {
  try {
    return Column::New(Env(), cudf::replace_nans(*this, replacement, mr));
  } catch (std::exception const& e) { NAPI_THROW(Napi::Error::New(Env(), e.what())); }
}

Napi::Value Column::replace_nulls(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  try {
    if (Column::IsInstance(info[0])) { return replace_nulls(*Column::Unwrap(args[0]), args[1]); }
    if (Scalar::IsInstance(info[0])) { return replace_nulls(*Scalar::Unwrap(args[0]), args[1]); }
    if (args[0].IsBoolean()) {
      cudf::replace_policy policy{static_cast<bool>(args[0])};
      return replace_nulls(policy, args[1]);
    }
  } catch (std::exception const& e) { throw Napi::Error::New(info.Env(), e.what()); }
  throw Napi::Error::New(info.Env(), "replace_nulls requires a Column, Scalar, or Boolean");
}

Napi::Value Column::replace_nans(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  try {
    if (Column::IsInstance(info[0])) { return replace_nans(*Column::Unwrap(args[0]), args[1]); }
    if (Scalar::IsInstance(info[0])) { return replace_nans(*Scalar::Unwrap(args[0]), args[1]); }
  } catch (std::exception const& e) { throw Napi::Error::New(info.Env(), e.what()); }
  throw Napi::Error::New(info.Env(), "replace_nans requires a Column or Scalar");
}

}  // namespace nv
