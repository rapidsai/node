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

#include <node_cudf/column.hpp>
#include <node_cudf/utilities/dtypes.hpp>

#include <node_rmm/memory_resource.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/scalar/scalar_factories.hpp>

namespace nv {

namespace {

Column::wrapper_t auto_binary_operation(
  Column const& lhs,
  Column const& rhs,
  cudf::binary_operator op,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) {
  return lhs.binary_operation(rhs, op, get_common_type(lhs.type(), rhs.type()), mr);
}

Column::wrapper_t auto_binary_operation(
  Column const& lhs,
  Scalar const& rhs,
  cudf::binary_operator op,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) {
  return lhs.binary_operation(rhs, op, get_common_type(lhs.type(), rhs.type()), mr);
}

}  // namespace

Column::wrapper_t Column::binary_operation(Column const& rhs,
                                           cudf::binary_operator op,
                                           cudf::type_id output_type,
                                           rmm::mr::device_memory_resource* mr) const {
  try {
    return Column::New(
      rhs.Env(), cudf::jit::binary_operation(*this, rhs, op, cudf::data_type{output_type}, mr));
  } catch (cudf::logic_error const& err) { NAPI_THROW(Napi::Error::New(rhs.Env(), err.what())); }
}

Column::wrapper_t Column::binary_operation(Scalar const& rhs,
                                           cudf::binary_operator op,
                                           cudf::type_id output_type,
                                           rmm::mr::device_memory_resource* mr) const {
  try {
    return Column::New(
      rhs.Env(), cudf::jit::binary_operation(*this, rhs, op, cudf::data_type{output_type}, mr));
  } catch (cudf::logic_error const& err) { NAPI_THROW(Napi::Error::New(rhs.Env(), err.what())); }
}

Column::wrapper_t Column::operator+(Column const& other) const { return add(other); }
Column::wrapper_t Column::operator+(Scalar const& other) const { return add(other); }

Column::wrapper_t Column::add(Column const& other, rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::ADD, mr);
}

Column::wrapper_t Column::add(Scalar const& other, rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::ADD, mr);
}

Column::wrapper_t Column::operator-(Column const& other) const { return sub(other); }
Column::wrapper_t Column::operator-(Scalar const& other) const { return sub(other); }

Column::wrapper_t Column::sub(Column const& other, rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::SUB, mr);
}

Column::wrapper_t Column::sub(Scalar const& other, rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::SUB, mr);
}

Column::wrapper_t Column::operator*(Column const& other) const { return mul(other); }
Column::wrapper_t Column::operator*(Scalar const& other) const { return mul(other); }

Column::wrapper_t Column::mul(Column const& other, rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::MUL, mr);
}

Column::wrapper_t Column::mul(Scalar const& other, rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::MUL, mr);
}

Column::wrapper_t Column::operator/(Column const& other) const { return div(other); }
Column::wrapper_t Column::operator/(Scalar const& other) const { return div(other); }

Column::wrapper_t Column::div(Column const& other, rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::DIV, mr);
}

Column::wrapper_t Column::div(Scalar const& other, rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::DIV, mr);
}

Column::wrapper_t Column::true_div(Column const& other, rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::TRUE_DIV, mr);
}

Column::wrapper_t Column::true_div(Scalar const& other, rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::TRUE_DIV, mr);
}

Column::wrapper_t Column::floor_div(Column const& other,
                                    rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::FLOOR_DIV, mr);
}

Column::wrapper_t Column::floor_div(Scalar const& other,
                                    rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::FLOOR_DIV, mr);
}

Column::wrapper_t Column::operator%(Column const& other) const { return mod(other); }
Column::wrapper_t Column::operator%(Scalar const& other) const { return mod(other); }

Column::wrapper_t Column::mod(Column const& other, rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::MOD, mr);
}

Column::wrapper_t Column::mod(Scalar const& other, rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::MOD, mr);
}

Column::wrapper_t Column::pow(Column const& other, rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::POW, mr);
}

Column::wrapper_t Column::pow(Scalar const& other, rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::POW, mr);
}

Column::wrapper_t Column::operator==(Column const& other) const { return eq(other); }
Column::wrapper_t Column::operator==(Scalar const& other) const { return eq(other); }

Column::wrapper_t Column::eq(Column const& other, rmm::mr::device_memory_resource* mr) const {
  return this->binary_operation(other, cudf::binary_operator::EQUAL, cudf::type_id::BOOL8, mr);
}

Column::wrapper_t Column::eq(Scalar const& other, rmm::mr::device_memory_resource* mr) const {
  return this->binary_operation(other, cudf::binary_operator::EQUAL, cudf::type_id::BOOL8, mr);
}

Column::wrapper_t Column::operator!=(Column const& other) const { return ne(other); }
Column::wrapper_t Column::operator!=(Scalar const& other) const { return ne(other); }

Column::wrapper_t Column::ne(Column const& other, rmm::mr::device_memory_resource* mr) const {
  return this->binary_operation(other, cudf::binary_operator::NOT_EQUAL, cudf::type_id::BOOL8, mr);
}

Column::wrapper_t Column::ne(Scalar const& other, rmm::mr::device_memory_resource* mr) const {
  return this->binary_operation(other, cudf::binary_operator::NOT_EQUAL, cudf::type_id::BOOL8, mr);
}

Column::wrapper_t Column::operator<(Column const& other) const { return lt(other); }
Column::wrapper_t Column::operator<(Scalar const& other) const { return lt(other); }

Column::wrapper_t Column::lt(Column const& other, rmm::mr::device_memory_resource* mr) const {
  return this->binary_operation(other, cudf::binary_operator::LESS, cudf::type_id::BOOL8, mr);
}

Column::wrapper_t Column::lt(Scalar const& other, rmm::mr::device_memory_resource* mr) const {
  return this->binary_operation(other, cudf::binary_operator::LESS, cudf::type_id::BOOL8, mr);
}

Column::wrapper_t Column::operator<=(Column const& other) const { return le(other); }
Column::wrapper_t Column::operator<=(Scalar const& other) const { return le(other); }

Column::wrapper_t Column::le(Column const& other, rmm::mr::device_memory_resource* mr) const {
  return this->binary_operation(other, cudf::binary_operator::LESS_EQUAL, cudf::type_id::BOOL8, mr);
}

Column::wrapper_t Column::le(Scalar const& other, rmm::mr::device_memory_resource* mr) const {
  return this->binary_operation(other, cudf::binary_operator::LESS_EQUAL, cudf::type_id::BOOL8, mr);
}

Column::wrapper_t Column::operator>(Column const& other) const { return gt(other); }
Column::wrapper_t Column::operator>(Scalar const& other) const { return gt(other); }

Column::wrapper_t Column::gt(Column const& other, rmm::mr::device_memory_resource* mr) const {
  return this->binary_operation(other, cudf::binary_operator::GREATER, cudf::type_id::BOOL8, mr);
}

Column::wrapper_t Column::gt(Scalar const& other, rmm::mr::device_memory_resource* mr) const {
  return this->binary_operation(other, cudf::binary_operator::GREATER, cudf::type_id::BOOL8, mr);
}

Column::wrapper_t Column::operator>=(Column const& other) const { return ge(other); }
Column::wrapper_t Column::operator>=(Scalar const& other) const { return ge(other); }

Column::wrapper_t Column::ge(Column const& other, rmm::mr::device_memory_resource* mr) const {
  return this->binary_operation(
    other, cudf::binary_operator::GREATER_EQUAL, cudf::type_id::BOOL8, mr);
}

Column::wrapper_t Column::ge(Scalar const& other, rmm::mr::device_memory_resource* mr) const {
  return this->binary_operation(
    other, cudf::binary_operator::GREATER_EQUAL, cudf::type_id::BOOL8, mr);
}

Column::wrapper_t Column::operator&(Column const& other) const { return bitwise_and(other); }
Column::wrapper_t Column::operator&(Scalar const& other) const { return bitwise_and(other); }

Column::wrapper_t Column::bitwise_and(Column const& other,
                                      rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::BITWISE_AND, mr);
}

Column::wrapper_t Column::bitwise_and(Scalar const& other,
                                      rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::BITWISE_AND, mr);
}

Column::wrapper_t Column::operator|(Column const& other) const { return bitwise_or(other); }
Column::wrapper_t Column::operator|(Scalar const& other) const { return bitwise_or(other); }

Column::wrapper_t Column::bitwise_or(Column const& other,
                                     rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::BITWISE_OR, mr);
}

Column::wrapper_t Column::bitwise_or(Scalar const& other,
                                     rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::BITWISE_OR, mr);
}

Column::wrapper_t Column::operator^(Column const& other) const { return bitwise_xor(other); }
Column::wrapper_t Column::operator^(Scalar const& other) const { return bitwise_xor(other); }

Column::wrapper_t Column::bitwise_xor(Column const& other,
                                      rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::BITWISE_XOR, mr);
}

Column::wrapper_t Column::bitwise_xor(Scalar const& other,
                                      rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::BITWISE_XOR, mr);
}

Column::wrapper_t Column::operator&&(Column const& other) const { return logical_and(other); }
Column::wrapper_t Column::operator&&(Scalar const& other) const { return logical_and(other); }

Column::wrapper_t Column::logical_and(Column const& other,
                                      rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::LOGICAL_AND, mr);
}

Column::wrapper_t Column::logical_and(Scalar const& other,
                                      rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::LOGICAL_AND, mr);
}

Column::wrapper_t Column::operator||(Column const& other) const { return logical_or(other); }
Column::wrapper_t Column::operator||(Scalar const& other) const { return logical_or(other); }

Column::wrapper_t Column::logical_or(Column const& other,
                                     rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::LOGICAL_OR, mr);
}

Column::wrapper_t Column::logical_or(Scalar const& other,
                                     rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::LOGICAL_OR, mr);
}

Column::wrapper_t Column::operator<<(Column const& other) const { return shift_left(other); }
Column::wrapper_t Column::operator<<(Scalar const& other) const { return shift_left(other); }

Column::wrapper_t Column::shift_left(Column const& other,
                                     rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::SHIFT_LEFT, mr);
}

Column::wrapper_t Column::shift_left(Scalar const& other,
                                     rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::SHIFT_LEFT, mr);
}

Column::wrapper_t Column::operator>>(Column const& other) const { return shift_right(other); }
Column::wrapper_t Column::operator>>(Scalar const& other) const { return shift_right(other); }

Column::wrapper_t Column::shift_right(Column const& other,
                                      rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::SHIFT_RIGHT, mr);
}

Column::wrapper_t Column::shift_right(Scalar const& other,
                                      rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::SHIFT_RIGHT, mr);
}

Column::wrapper_t Column::shift_right_unsigned(Column const& other,
                                               rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::SHIFT_RIGHT_UNSIGNED, mr);
}

Column::wrapper_t Column::shift_right_unsigned(Scalar const& other,
                                               rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::SHIFT_RIGHT_UNSIGNED, mr);
}

Column::wrapper_t Column::log_base(Column const& other, rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::LOG_BASE, mr);
}

Column::wrapper_t Column::log_base(Scalar const& other, rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::LOG_BASE, mr);
}

Column::wrapper_t Column::atan2(Column const& other, rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::ATAN2, mr);
}

Column::wrapper_t Column::atan2(Scalar const& other, rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::ATAN2, mr);
}

Column::wrapper_t Column::null_equals(Column const& other,
                                      rmm::mr::device_memory_resource* mr) const {
  return this->binary_operation(
    other, cudf::binary_operator::NULL_EQUALS, cudf::type_id::BOOL8, mr);
}

Column::wrapper_t Column::null_equals(Scalar const& other,
                                      rmm::mr::device_memory_resource* mr) const {
  return this->binary_operation(
    other, cudf::binary_operator::NULL_EQUALS, cudf::type_id::BOOL8, mr);
}

Column::wrapper_t Column::null_max(Column const& other, rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::NULL_MAX, mr);
}

Column::wrapper_t Column::null_max(Scalar const& other, rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::NULL_MAX, mr);
}

Column::wrapper_t Column::null_min(Column const& other, rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::NULL_MIN, mr);
}

Column::wrapper_t Column::null_min(Scalar const& other, rmm::mr::device_memory_resource* mr) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::NULL_MIN, mr);
}

// Private (JS-facing) impls

Napi::Value Column::add(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  rmm::mr::device_memory_resource* mr{NapiToCPP(info[1])};
  if (Column::IsInstance(rhs)) { return add(*Column::Unwrap(rhs.ToObject()), mr); }
  if (Scalar::IsInstance(rhs)) { return add(*Scalar::Unwrap(rhs.ToObject()), mr); }
  if (rhs.IsBigInt()) { return add(Scalar::New(info.Env(), rhs.As<Napi::BigInt>()), mr); }
  if (rhs.IsNumber()) { return add(Scalar::New(info.Env(), rhs.As<Napi::Number>()), mr); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.add expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::sub(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  rmm::mr::device_memory_resource* mr{NapiToCPP(info[1])};
  if (Column::IsInstance(rhs)) { return sub(*Column::Unwrap(rhs.ToObject()), mr); }
  if (Scalar::IsInstance(rhs)) { return sub(*Scalar::Unwrap(rhs.ToObject()), mr); }
  if (rhs.IsBigInt()) { return sub(Scalar::New(info.Env(), rhs.As<Napi::BigInt>()), mr); }
  if (rhs.IsNumber()) { return sub(Scalar::New(info.Env(), rhs.As<Napi::Number>()), mr); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.sub expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::mul(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  rmm::mr::device_memory_resource* mr{NapiToCPP(info[1])};
  if (Column::IsInstance(rhs)) { return mul(*Column::Unwrap(rhs.ToObject()), mr); }
  if (Scalar::IsInstance(rhs)) { return mul(*Scalar::Unwrap(rhs.ToObject()), mr); }
  if (rhs.IsBigInt()) { return mul(Scalar::New(info.Env(), rhs.As<Napi::BigInt>()), mr); }
  if (rhs.IsNumber()) { return mul(Scalar::New(info.Env(), rhs.As<Napi::Number>()), mr); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.mul expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::div(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  rmm::mr::device_memory_resource* mr{NapiToCPP(info[1])};
  if (Column::IsInstance(rhs)) { return div(*Column::Unwrap(rhs.ToObject()), mr); }
  if (Scalar::IsInstance(rhs)) { return div(*Scalar::Unwrap(rhs.ToObject()), mr); }
  if (rhs.IsBigInt()) { return div(Scalar::New(info.Env(), rhs.As<Napi::BigInt>()), mr); }
  if (rhs.IsNumber()) { return div(Scalar::New(info.Env(), rhs.As<Napi::Number>()), mr); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.div expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::true_div(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  rmm::mr::device_memory_resource* mr{NapiToCPP(info[1])};
  if (Column::IsInstance(rhs)) { return true_div(*Column::Unwrap(rhs.ToObject()), mr); }
  if (Scalar::IsInstance(rhs)) { return true_div(*Scalar::Unwrap(rhs.ToObject()), mr); }
  if (rhs.IsBigInt()) { return true_div(Scalar::New(info.Env(), rhs.As<Napi::BigInt>()), mr); }
  if (rhs.IsNumber()) { return true_div(Scalar::New(info.Env(), rhs.As<Napi::Number>()), mr); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.true_div expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::floor_div(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  rmm::mr::device_memory_resource* mr{NapiToCPP(info[1])};
  if (Column::IsInstance(rhs)) { return floor_div(*Column::Unwrap(rhs.ToObject()), mr); }
  if (Scalar::IsInstance(rhs)) { return floor_div(*Scalar::Unwrap(rhs.ToObject()), mr); }
  if (rhs.IsBigInt()) { return floor_div(Scalar::New(info.Env(), rhs.As<Napi::BigInt>()), mr); }
  if (rhs.IsNumber()) { return floor_div(Scalar::New(info.Env(), rhs.As<Napi::Number>()), mr); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.floor_div expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::mod(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  rmm::mr::device_memory_resource* mr{NapiToCPP(info[1])};
  if (Column::IsInstance(rhs)) { return mod(*Column::Unwrap(rhs.ToObject()), mr); }
  if (Scalar::IsInstance(rhs)) { return mod(*Scalar::Unwrap(rhs.ToObject()), mr); }
  if (rhs.IsBigInt()) { return mod(Scalar::New(info.Env(), rhs.As<Napi::BigInt>()), mr); }
  if (rhs.IsNumber()) { return mod(Scalar::New(info.Env(), rhs.As<Napi::Number>()), mr); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.mod expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::pow(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  rmm::mr::device_memory_resource* mr{NapiToCPP(info[1])};
  if (Column::IsInstance(rhs)) { return pow(*Column::Unwrap(rhs.ToObject()), mr); }
  if (Scalar::IsInstance(rhs)) { return pow(*Scalar::Unwrap(rhs.ToObject()), mr); }
  if (rhs.IsBigInt()) { return pow(Scalar::New(info.Env(), rhs.As<Napi::BigInt>()), mr); }
  if (rhs.IsNumber()) { return pow(Scalar::New(info.Env(), rhs.As<Napi::Number>()), mr); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.pow expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::eq(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  rmm::mr::device_memory_resource* mr{NapiToCPP(info[1])};
  if (Column::IsInstance(rhs)) { return eq(*Column::Unwrap(rhs.ToObject()), mr); }
  if (Scalar::IsInstance(rhs)) { return eq(*Scalar::Unwrap(rhs.ToObject()), mr); }
  if (rhs.IsBigInt()) { return eq(Scalar::New(info.Env(), rhs.As<Napi::BigInt>()), mr); }
  if (rhs.IsNumber()) { return eq(Scalar::New(info.Env(), rhs.As<Napi::Number>()), mr); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.eq expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::ne(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  rmm::mr::device_memory_resource* mr{NapiToCPP(info[1])};
  if (Column::IsInstance(rhs)) { return ne(*Column::Unwrap(rhs.ToObject()), mr); }
  if (Scalar::IsInstance(rhs)) { return ne(*Scalar::Unwrap(rhs.ToObject()), mr); }
  if (rhs.IsBigInt()) { return ne(Scalar::New(info.Env(), rhs.As<Napi::BigInt>()), mr); }
  if (rhs.IsNumber()) { return ne(Scalar::New(info.Env(), rhs.As<Napi::Number>()), mr); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.ne expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::lt(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  rmm::mr::device_memory_resource* mr{NapiToCPP(info[1])};
  if (Column::IsInstance(rhs)) { return lt(*Column::Unwrap(rhs.ToObject()), mr); }
  if (Scalar::IsInstance(rhs)) { return lt(*Scalar::Unwrap(rhs.ToObject()), mr); }
  if (rhs.IsBigInt()) { return lt(Scalar::New(info.Env(), rhs.As<Napi::BigInt>()), mr); }
  if (rhs.IsNumber()) { return lt(Scalar::New(info.Env(), rhs.As<Napi::Number>()), mr); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.lt expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::le(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  rmm::mr::device_memory_resource* mr{NapiToCPP(info[1])};
  if (Column::IsInstance(rhs)) { return le(*Column::Unwrap(rhs.ToObject()), mr); }
  if (Scalar::IsInstance(rhs)) { return le(*Scalar::Unwrap(rhs.ToObject()), mr); }
  if (rhs.IsBigInt()) { return le(Scalar::New(info.Env(), rhs.As<Napi::BigInt>()), mr); }
  if (rhs.IsNumber()) { return le(Scalar::New(info.Env(), rhs.As<Napi::Number>()), mr); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.le expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::gt(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  rmm::mr::device_memory_resource* mr{NapiToCPP(info[1])};
  if (Column::IsInstance(rhs)) { return gt(*Column::Unwrap(rhs.ToObject()), mr); }
  if (Scalar::IsInstance(rhs)) { return gt(*Scalar::Unwrap(rhs.ToObject()), mr); }
  if (rhs.IsBigInt()) { return gt(Scalar::New(info.Env(), rhs.As<Napi::BigInt>()), mr); }
  if (rhs.IsNumber()) { return gt(Scalar::New(info.Env(), rhs.As<Napi::Number>()), mr); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.gt expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::ge(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  rmm::mr::device_memory_resource* mr{NapiToCPP(info[1])};
  if (Column::IsInstance(rhs)) { return ge(*Column::Unwrap(rhs.ToObject()), mr); }
  if (Scalar::IsInstance(rhs)) { return ge(*Scalar::Unwrap(rhs.ToObject()), mr); }
  if (rhs.IsBigInt()) { return ge(Scalar::New(info.Env(), rhs.As<Napi::BigInt>()), mr); }
  if (rhs.IsNumber()) { return ge(Scalar::New(info.Env(), rhs.As<Napi::Number>()), mr); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.ge expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::bitwise_and(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  rmm::mr::device_memory_resource* mr{NapiToCPP(info[1])};
  if (Column::IsInstance(rhs)) { return bitwise_and(*Column::Unwrap(rhs.ToObject()), mr); }
  if (Scalar::IsInstance(rhs)) { return bitwise_and(*Scalar::Unwrap(rhs.ToObject()), mr); }
  if (rhs.IsBigInt()) { return bitwise_and(Scalar::New(info.Env(), rhs, type()), mr); }
  if (rhs.IsNumber()) { return bitwise_and(Scalar::New(info.Env(), rhs, type()), mr); }
  NAPI_THROW(Napi::Error::New(info.Env(),
                              "Column.bitwise_and expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::bitwise_or(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  rmm::mr::device_memory_resource* mr{NapiToCPP(info[1])};
  if (Column::IsInstance(rhs)) { return bitwise_or(*Column::Unwrap(rhs.ToObject()), mr); }
  if (Scalar::IsInstance(rhs)) { return bitwise_or(*Scalar::Unwrap(rhs.ToObject()), mr); }
  if (rhs.IsBigInt()) { return bitwise_or(Scalar::New(info.Env(), rhs, type()), mr); }
  if (rhs.IsNumber()) { return bitwise_or(Scalar::New(info.Env(), rhs, type()), mr); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.bitwise_or expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::bitwise_xor(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  rmm::mr::device_memory_resource* mr{NapiToCPP(info[1])};
  if (Column::IsInstance(rhs)) { return bitwise_xor(*Column::Unwrap(rhs.ToObject()), mr); }
  if (Scalar::IsInstance(rhs)) { return bitwise_xor(*Scalar::Unwrap(rhs.ToObject()), mr); }
  if (rhs.IsBigInt()) { return bitwise_xor(Scalar::New(info.Env(), rhs, type()), mr); }
  if (rhs.IsNumber()) { return bitwise_xor(Scalar::New(info.Env(), rhs, type()), mr); }
  NAPI_THROW(Napi::Error::New(info.Env(),
                              "Column.bitwise_xor expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::logical_and(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  rmm::mr::device_memory_resource* mr{NapiToCPP(info[1])};
  if (Column::IsInstance(rhs)) { return logical_and(*Column::Unwrap(rhs.ToObject()), mr); }
  if (Scalar::IsInstance(rhs)) { return logical_and(*Scalar::Unwrap(rhs.ToObject()), mr); }
  if (rhs.IsBigInt()) { return logical_and(Scalar::New(info.Env(), rhs.As<Napi::BigInt>()), mr); }
  if (rhs.IsNumber()) { return logical_and(Scalar::New(info.Env(), rhs.As<Napi::Number>()), mr); }
  NAPI_THROW(Napi::Error::New(info.Env(),
                              "Column.logical_and expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::logical_or(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  rmm::mr::device_memory_resource* mr{NapiToCPP(info[1])};
  if (Column::IsInstance(rhs)) { return logical_or(*Column::Unwrap(rhs.ToObject()), mr); }
  if (Scalar::IsInstance(rhs)) { return logical_or(*Scalar::Unwrap(rhs.ToObject()), mr); }
  if (rhs.IsBigInt()) { return logical_or(Scalar::New(info.Env(), rhs.As<Napi::BigInt>()), mr); }
  if (rhs.IsNumber()) { return logical_or(Scalar::New(info.Env(), rhs.As<Napi::Number>()), mr); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.logical_or expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::shift_left(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  rmm::mr::device_memory_resource* mr{NapiToCPP(info[1])};
  if (Column::IsInstance(rhs)) { return shift_left(*Column::Unwrap(rhs.ToObject()), mr); }
  if (Scalar::IsInstance(rhs)) { return shift_left(*Scalar::Unwrap(rhs.ToObject()), mr); }
  if (rhs.IsBigInt()) { return shift_left(Scalar::New(info.Env(), rhs, type()), mr); }
  if (rhs.IsNumber()) { return shift_left(Scalar::New(info.Env(), rhs, type()), mr); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.shift_left expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::shift_right(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  rmm::mr::device_memory_resource* mr{NapiToCPP(info[1])};
  if (Column::IsInstance(rhs)) { return shift_right(*Column::Unwrap(rhs.ToObject()), mr); }
  if (Scalar::IsInstance(rhs)) { return shift_right(*Scalar::Unwrap(rhs.ToObject()), mr); }
  if (rhs.IsBigInt()) { return shift_right(Scalar::New(info.Env(), rhs, type()), mr); }
  if (rhs.IsNumber()) { return shift_right(Scalar::New(info.Env(), rhs, type()), mr); }
  NAPI_THROW(Napi::Error::New(info.Env(),
                              "Column.shift_right expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::shift_right_unsigned(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  rmm::mr::device_memory_resource* mr{NapiToCPP(info[1])};
  if (Column::IsInstance(rhs)) { return shift_right_unsigned(*Column::Unwrap(rhs.ToObject()), mr); }
  if (Scalar::IsInstance(rhs)) { return shift_right_unsigned(*Scalar::Unwrap(rhs.ToObject()), mr); }
  if (rhs.IsBigInt()) { return shift_right_unsigned(Scalar::New(info.Env(), rhs, type()), mr); }
  if (rhs.IsNumber()) { return shift_right_unsigned(Scalar::New(info.Env(), rhs, type()), mr); }
  NAPI_THROW(Napi::Error::New(
    info.Env(), "Column.shift_right_unsigned expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::log_base(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  rmm::mr::device_memory_resource* mr{NapiToCPP(info[1])};
  if (Column::IsInstance(rhs)) { return log_base(*Column::Unwrap(rhs.ToObject()), mr); }
  if (Scalar::IsInstance(rhs)) { return log_base(*Scalar::Unwrap(rhs.ToObject()), mr); }
  if (rhs.IsBigInt()) { return log_base(Scalar::New(info.Env(), rhs.As<Napi::BigInt>()), mr); }
  if (rhs.IsNumber()) { return log_base(Scalar::New(info.Env(), rhs.As<Napi::Number>()), mr); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.log_base expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::atan2(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  rmm::mr::device_memory_resource* mr{NapiToCPP(info[1])};
  if (Column::IsInstance(rhs)) { return atan2(*Column::Unwrap(rhs.ToObject()), mr); }
  if (Scalar::IsInstance(rhs)) { return atan2(*Scalar::Unwrap(rhs.ToObject()), mr); }
  if (rhs.IsBigInt()) { return atan2(Scalar::New(info.Env(), rhs.As<Napi::BigInt>()), mr); }
  if (rhs.IsNumber()) { return atan2(Scalar::New(info.Env(), rhs.As<Napi::Number>()), mr); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.atan2 expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::null_equals(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  rmm::mr::device_memory_resource* mr{NapiToCPP(info[1])};
  if (Column::IsInstance(rhs)) { return null_equals(*Column::Unwrap(rhs.ToObject()), mr); }
  if (Scalar::IsInstance(rhs)) { return null_equals(*Scalar::Unwrap(rhs.ToObject()), mr); }
  if (rhs.IsBigInt()) { return null_equals(Scalar::New(info.Env(), rhs.As<Napi::BigInt>()), mr); }
  if (rhs.IsNumber()) { return null_equals(Scalar::New(info.Env(), rhs.As<Napi::Number>()), mr); }
  NAPI_THROW(Napi::Error::New(info.Env(),
                              "Column.null_equals expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::null_max(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  rmm::mr::device_memory_resource* mr{NapiToCPP(info[1])};
  if (Column::IsInstance(rhs)) { return null_max(*Column::Unwrap(rhs.ToObject()), mr); }
  if (Scalar::IsInstance(rhs)) { return null_max(*Scalar::Unwrap(rhs.ToObject()), mr); }
  if (rhs.IsBigInt()) { return null_max(Scalar::New(info.Env(), rhs.As<Napi::BigInt>()), mr); }
  if (rhs.IsNumber()) { return null_max(Scalar::New(info.Env(), rhs.As<Napi::Number>()), mr); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.null_max expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::null_min(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  rmm::mr::device_memory_resource* mr{NapiToCPP(info[1])};
  if (Column::IsInstance(rhs)) { return null_min(*Column::Unwrap(rhs.ToObject()), mr); }
  if (Scalar::IsInstance(rhs)) { return null_min(*Scalar::Unwrap(rhs.ToObject()), mr); }
  if (rhs.IsBigInt()) { return null_min(Scalar::New(info.Env(), rhs.As<Napi::BigInt>()), mr); }
  if (rhs.IsNumber()) { return null_min(Scalar::New(info.Env(), rhs.As<Napi::Number>()), mr); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.null_min expects a Column, Scalar, bigint, or number."));
}

}  // namespace nv
