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

namespace {

inline rmm::mr::device_memory_resource* get_mr(Napi::Value const& arg) {
  return MemoryResource::is_instance(arg) ? *MemoryResource::Unwrap(arg.ToObject())
                                          : rmm::mr::get_current_device_resource();
}

}  // namespace

ObjectUnwrap<Column> Column::cast(cudf::data_type out_type,
                                  rmm::mr::device_memory_resource* mr) const {
  return Column::New(cudf::cast(*this, out_type, mr));
}

ObjectUnwrap<Column> Column::is_null(rmm::mr::device_memory_resource* mr) const {
  return Column::New(cudf::is_null(*this, mr));
}

ObjectUnwrap<Column> Column::is_valid(rmm::mr::device_memory_resource* mr) const {
  return Column::New(cudf::is_valid(*this, mr));
}

ObjectUnwrap<Column> Column::is_nan(rmm::mr::device_memory_resource* mr) const {
  return Column::New(cudf::is_nan(*this, mr));
}

ObjectUnwrap<Column> Column::is_not_nan(rmm::mr::device_memory_resource* mr) const {
  return Column::New(cudf::is_not_nan(*this, mr));
}

ObjectUnwrap<Column> Column::unary_operation(cudf::unary_operator op,
                                             rmm::mr::device_memory_resource* mr) const {
  return Column::New(cudf::unary_operation(*this, op, mr));
}

Napi::Value Column::cast(Napi::CallbackInfo const& info) {
  switch (info.Length()) {
    case 1:
    case 2: return cast(NapiToCPP{info[0]}, get_mr(info[1]));
    default:
      NODE_CUDF_THROW("Column cast expects a DataType and optional MemoryResource", info.Env());
  }
}

Napi::Value Column::is_null(Napi::CallbackInfo const& info) { return is_null(get_mr(info[0])); }

Napi::Value Column::is_valid(Napi::CallbackInfo const& info) { return is_valid(get_mr(info[0])); }

Napi::Value Column::is_nan(Napi::CallbackInfo const& info) { return is_nan(get_mr(info[0])); }

Napi::Value Column::is_not_nan(Napi::CallbackInfo const& info) {
  return is_not_nan(get_mr(info[0]));
}

Napi::Value Column::sin(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::SIN, get_mr(info[0]));
}

Napi::Value Column::cos(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::COS, get_mr(info[0]));
}

Napi::Value Column::tan(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::TAN, get_mr(info[0]));
}

Napi::Value Column::arcsin(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::ARCSIN, get_mr(info[0]));
}

Napi::Value Column::arccos(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::ARCCOS, get_mr(info[0]));
}

Napi::Value Column::arctan(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::ARCTAN, get_mr(info[0]));
}

Napi::Value Column::sinh(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::SINH, get_mr(info[0]));
}

Napi::Value Column::cosh(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::COSH, get_mr(info[0]));
}

Napi::Value Column::tanh(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::TANH, get_mr(info[0]));
}

Napi::Value Column::arcsinh(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::ARCSINH, get_mr(info[0]));
}

Napi::Value Column::arccosh(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::ARCCOSH, get_mr(info[0]));
}

Napi::Value Column::arctanh(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::ARCTANH, get_mr(info[0]));
}

Napi::Value Column::exp(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::EXP, get_mr(info[0]));
}

Napi::Value Column::log(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::LOG, get_mr(info[0]));
}

Napi::Value Column::sqrt(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::SQRT, get_mr(info[0]));
}

Napi::Value Column::cbrt(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::CBRT, get_mr(info[0]));
}

Napi::Value Column::ceil(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::CEIL, get_mr(info[0]));
}

Napi::Value Column::floor(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::FLOOR, get_mr(info[0]));
}

Napi::Value Column::abs(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::ABS, get_mr(info[0]));
}

Napi::Value Column::rint(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::RINT, get_mr(info[0]));
}

Napi::Value Column::bit_invert(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::BIT_INVERT, get_mr(info[0]));
}

Napi::Value Column::unary_not(Napi::CallbackInfo const& info) {
  return unary_operation(cudf::unary_operator::NOT, get_mr(info[0]));
}

}  // namespace nv
