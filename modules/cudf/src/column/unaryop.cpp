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
#include <node_cudf/utilities/napi_to_cpp.hpp>

#include <node_rmm/memory_resource.hpp>

#include <cudf/unary.hpp>

#include <rmm/mr/device/per_device_resource.hpp>

namespace nv {

Column::wrapper_t Column::cast(cudf::data_type out_type,
                               rmm::mr::device_memory_resource* mr) const {
  try {
    return Column::New(Env(), cudf::cast(*this, out_type, mr));
  } catch (cudf::logic_error const& err) { NAPI_THROW(Napi::Error::New(Env(), err.what())); }
}

Column::wrapper_t Column::is_null(rmm::mr::device_memory_resource* mr) const {
  try {
    return Column::New(Env(), cudf::is_null(*this, mr));
  } catch (cudf::logic_error const& err) { NAPI_THROW(Napi::Error::New(Env(), err.what())); }
}

Column::wrapper_t Column::is_valid(rmm::mr::device_memory_resource* mr) const {
  try {
    return Column::New(Env(), cudf::is_valid(*this, mr));
  } catch (cudf::logic_error const& err) { NAPI_THROW(Napi::Error::New(Env(), err.what())); }
}

Column::wrapper_t Column::is_nan(rmm::mr::device_memory_resource* mr) const {
  try {
    return Column::New(Env(), cudf::is_nan(*this, mr));
  } catch (cudf::logic_error const& err) { NAPI_THROW(Napi::Error::New(Env(), err.what())); }
}

Column::wrapper_t Column::is_not_nan(rmm::mr::device_memory_resource* mr) const {
  try {
    return Column::New(Env(), cudf::is_not_nan(*this, mr));
  } catch (cudf::logic_error const& err) { NAPI_THROW(Napi::Error::New(Env(), err.what())); }
}

Column::wrapper_t Column::unary_operation(cudf::unary_operator op,
                                          rmm::mr::device_memory_resource* mr) const {
  try {
    return Column::New(Env(), cudf::unary_operation(*this, op, mr));
  } catch (cudf::logic_error const& err) { NAPI_THROW(Napi::Error::New(Env(), err.what())); }
}

Napi::Value Column::cast(Napi::CallbackInfo const& info) {
  if (info.Length() < 1) {
    NODE_CUDF_THROW("Column cast expects a DataType and optional MemoryResource", info.Env());
  }
  return cast(NapiToCPP{info[0]}, NapiToCPP(info[1]));
}

Napi::Value Column::is_null(Napi::CallbackInfo const& info) {
  return is_null(NapiToCPP(info[0]).operator rmm::mr::device_memory_resource*());
}

Napi::Value Column::is_valid(Napi::CallbackInfo const& info) {
  return is_valid(NapiToCPP(info[0]).operator rmm::mr::device_memory_resource*());
}

Napi::Value Column::is_nan(Napi::CallbackInfo const& info) {
  return is_nan(NapiToCPP(info[0]).operator rmm::mr::device_memory_resource*());
}

Napi::Value Column::is_not_nan(Napi::CallbackInfo const& info) {
  return is_not_nan(NapiToCPP(info[0]).operator rmm::mr::device_memory_resource*());
}

Napi::Value Column::sin(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::SIN, NapiToCPP(info[0]));
}

Napi::Value Column::cos(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::COS, NapiToCPP(info[0]));
}

Napi::Value Column::tan(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::TAN, NapiToCPP(info[0]));
}

Napi::Value Column::arcsin(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::ARCSIN, NapiToCPP(info[0]));
}

Napi::Value Column::arccos(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::ARCCOS, NapiToCPP(info[0]));
}

Napi::Value Column::arctan(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::ARCTAN, NapiToCPP(info[0]));
}

Napi::Value Column::sinh(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::SINH, NapiToCPP(info[0]));
}

Napi::Value Column::cosh(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::COSH, NapiToCPP(info[0]));
}

Napi::Value Column::tanh(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::TANH, NapiToCPP(info[0]));
}

Napi::Value Column::arcsinh(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::ARCSINH, NapiToCPP(info[0]));
}

Napi::Value Column::arccosh(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::ARCCOSH, NapiToCPP(info[0]));
}

Napi::Value Column::arctanh(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::ARCTANH, NapiToCPP(info[0]));
}

Napi::Value Column::exp(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::EXP, NapiToCPP(info[0]));
}

Napi::Value Column::log(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::LOG, NapiToCPP(info[0]));
}

Napi::Value Column::sqrt(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::SQRT, NapiToCPP(info[0]));
}

Napi::Value Column::cbrt(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::CBRT, NapiToCPP(info[0]));
}

Napi::Value Column::ceil(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::CEIL, NapiToCPP(info[0]));
}

Napi::Value Column::floor(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::FLOOR, NapiToCPP(info[0]));
}

Napi::Value Column::abs(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::ABS, NapiToCPP(info[0]));
}

Napi::Value Column::rint(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::RINT, NapiToCPP(info[0]));
}

Napi::Value Column::bit_invert(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::BIT_INVERT, NapiToCPP(info[0]));
}

Napi::Value Column::unary_not(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::NOT, NapiToCPP(info[0]));
}

}  // namespace nv
