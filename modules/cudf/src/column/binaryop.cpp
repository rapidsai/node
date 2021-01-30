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

#include <cudf/binaryop.hpp>
#include <cudf/scalar/scalar_factories.hpp>

namespace nv {

namespace {

template <typename _RHS>
struct get_output_type {
  template <
    typename LHS,
    typename RHS                                          = _RHS,
    typename std::enable_if_t<(std::is_convertible<LHS, RHS>::value && cudf::is_numeric<LHS>() &&
                               cudf::is_numeric<RHS>())>* = nullptr>
  cudf::type_id operator()(cudf::data_type const& lhs, cudf::data_type const& rhs) {
    return cudf::type_to_id<std::common_type_t<LHS, RHS>>();
  }
  template <
    typename LHS,
    typename RHS                                           = _RHS,
    typename std::enable_if_t<!(std::is_convertible<LHS, RHS>::value && cudf::is_numeric<LHS>() &&
                                cudf::is_numeric<RHS>())>* = nullptr>
  cudf::type_id operator()(cudf::data_type const& lhs, cudf::data_type const& rhs) {
    auto lhs_name = cudf::type_dispatcher(lhs, cudf::type_to_name{});
    auto rhs_name = cudf::type_dispatcher(rhs, cudf::type_to_name{});
    CUDF_FAIL("Cannot determine a logical common type between " + lhs_name + " and " + rhs_name);
  }
};

struct dispatch_get_output_type {
  template <typename RHS>
  cudf::type_id operator()(cudf::data_type const& lhs, cudf::data_type const& rhs) {
    return cudf::type_dispatcher(lhs, get_output_type<RHS>{}, lhs, rhs);
  }
};

cudf::type_id get_common_type(cudf::data_type const& lhs, cudf::data_type const& rhs) {
  return cudf::type_dispatcher(rhs, dispatch_get_output_type{}, lhs, rhs);
}

ObjectUnwrap<Column> auto_binary_operation(Column const& lhs,
                                           Column const& rhs,
                                           cudf::binary_operator op) {
  try {
    return lhs.binary_operation(rhs, op, get_common_type(lhs.type(), rhs.type()));
  } catch (cudf::logic_error const& err) { NAPI_THROW(Napi::Error::New(rhs.Env(), err.what())); }
}

ObjectUnwrap<Column> auto_binary_operation(Column const& lhs,
                                           Scalar const& rhs,
                                           cudf::binary_operator op) {
  try {
    return lhs.binary_operation(rhs, op, get_common_type(lhs.type(), rhs.type()));
  } catch (cudf::logic_error const& err) { NAPI_THROW(Napi::Error::New(rhs.Env(), err.what())); }
}

}  // namespace

ObjectUnwrap<Column> Column::binary_operation(Column const& rhs,
                                              cudf::binary_operator op,
                                              cudf::type_id output_type) const {
  return Column::New(cudf::binary_operation(*this, rhs, op, cudf::data_type{output_type}));
}

ObjectUnwrap<Column> Column::binary_operation(Scalar const& rhs,
                                              cudf::binary_operator op,
                                              cudf::type_id output_type) const {
  return Column::New(cudf::binary_operation(*this, rhs, op, cudf::data_type{output_type}));
}

ObjectUnwrap<Column> Column::operator+(Column const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::ADD);
}

ObjectUnwrap<Column> Column::operator+(Scalar const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::ADD);
}

ObjectUnwrap<Column> Column::operator-(Column const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::SUB);
}

ObjectUnwrap<Column> Column::operator-(Scalar const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::SUB);
}

ObjectUnwrap<Column> Column::operator*(Column const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::MUL);
}

ObjectUnwrap<Column> Column::operator*(Scalar const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::MUL);
}

ObjectUnwrap<Column> Column::operator/(Column const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::DIV);
}

ObjectUnwrap<Column> Column::operator/(Scalar const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::DIV);
}

ObjectUnwrap<Column> Column::true_div(Column const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::TRUE_DIV);
}

ObjectUnwrap<Column> Column::true_div(Scalar const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::TRUE_DIV);
}

ObjectUnwrap<Column> Column::floor_div(Column const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::FLOOR_DIV);
  // return this->binary_operation(
  //   other,
  //   cudf::binary_operator::FLOOR_DIV,
  //   type().id() == cudf::type_id::FLOAT32 ? cudf::type_id::FLOAT32 : cudf::type_id::FLOAT64);
}

ObjectUnwrap<Column> Column::floor_div(Scalar const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::FLOOR_DIV);
  // return this->binary_operation(
  //   other,
  //   cudf::binary_operator::FLOOR_DIV,
  //   type().id() == cudf::type_id::FLOAT32 ? cudf::type_id::FLOAT32 : cudf::type_id::FLOAT64);
}

ObjectUnwrap<Column> Column::operator%(Column const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::MOD);
  // return this->binary_operation(
  //   other,
  //   cudf::binary_operator::MOD,
  //   type().id() == cudf::type_id::FLOAT32 ? cudf::type_id::FLOAT32 : cudf::type_id::FLOAT64);
}

ObjectUnwrap<Column> Column::operator%(Scalar const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::MOD);
  // return this->binary_operation(
  //   other,
  //   cudf::binary_operator::MOD,
  //   type().id() == cudf::type_id::FLOAT32 ? cudf::type_id::FLOAT32 : cudf::type_id::FLOAT64);
}

ObjectUnwrap<Column> Column::pow(Column const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::POW);
}

ObjectUnwrap<Column> Column::pow(Scalar const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::POW);
}

ObjectUnwrap<Column> Column::operator==(Column const& other) const {
  return this->binary_operation(other, cudf::binary_operator::EQUAL, cudf::type_id::BOOL8);
}

ObjectUnwrap<Column> Column::operator==(Scalar const& other) const {
  return this->binary_operation(other, cudf::binary_operator::EQUAL, cudf::type_id::BOOL8);
}

ObjectUnwrap<Column> Column::operator!=(Column const& other) const {
  return this->binary_operation(other, cudf::binary_operator::NOT_EQUAL, cudf::type_id::BOOL8);
}

ObjectUnwrap<Column> Column::operator!=(Scalar const& other) const {
  return this->binary_operation(other, cudf::binary_operator::NOT_EQUAL, cudf::type_id::BOOL8);
}

ObjectUnwrap<Column> Column::operator<(Column const& other) const {
  return this->binary_operation(other, cudf::binary_operator::LESS, cudf::type_id::BOOL8);
}

ObjectUnwrap<Column> Column::operator<(Scalar const& other) const {
  return this->binary_operation(other, cudf::binary_operator::LESS, cudf::type_id::BOOL8);
}

ObjectUnwrap<Column> Column::operator<=(Column const& other) const {
  return this->binary_operation(other, cudf::binary_operator::LESS_EQUAL, cudf::type_id::BOOL8);
}

ObjectUnwrap<Column> Column::operator<=(Scalar const& other) const {
  return this->binary_operation(other, cudf::binary_operator::LESS_EQUAL, cudf::type_id::BOOL8);
}

ObjectUnwrap<Column> Column::operator>(Column const& other) const {
  return this->binary_operation(other, cudf::binary_operator::GREATER, cudf::type_id::BOOL8);
}

ObjectUnwrap<Column> Column::operator>(Scalar const& other) const {
  return this->binary_operation(other, cudf::binary_operator::GREATER, cudf::type_id::BOOL8);
}

ObjectUnwrap<Column> Column::operator>=(Column const& other) const {
  return this->binary_operation(other, cudf::binary_operator::GREATER_EQUAL, cudf::type_id::BOOL8);
}

ObjectUnwrap<Column> Column::operator>=(Scalar const& other) const {
  return this->binary_operation(other, cudf::binary_operator::GREATER_EQUAL, cudf::type_id::BOOL8);
}

ObjectUnwrap<Column> Column::operator&(Column const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::BITWISE_AND);
}

ObjectUnwrap<Column> Column::operator&(Scalar const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::BITWISE_AND);
}

ObjectUnwrap<Column> Column::operator|(Column const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::BITWISE_OR);
}

ObjectUnwrap<Column> Column::operator|(Scalar const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::BITWISE_OR);
}

ObjectUnwrap<Column> Column::operator^(Column const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::BITWISE_XOR);
}

ObjectUnwrap<Column> Column::operator^(Scalar const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::BITWISE_XOR);
}

ObjectUnwrap<Column> Column::operator&&(Column const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::LOGICAL_AND);
}

ObjectUnwrap<Column> Column::operator&&(Scalar const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::LOGICAL_AND);
}

ObjectUnwrap<Column> Column::operator||(Column const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::LOGICAL_OR);
}

ObjectUnwrap<Column> Column::operator||(Scalar const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::LOGICAL_OR);
}

ObjectUnwrap<Column> Column::coalesce(Column const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::COALESCE);
}

ObjectUnwrap<Column> Column::coalesce(Scalar const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::COALESCE);
}

ObjectUnwrap<Column> Column::operator<<(Column const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::SHIFT_LEFT);
}

ObjectUnwrap<Column> Column::operator<<(Scalar const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::SHIFT_LEFT);
}

ObjectUnwrap<Column> Column::operator>>(Column const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::SHIFT_RIGHT);
}

ObjectUnwrap<Column> Column::operator>>(Scalar const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::SHIFT_RIGHT);
}

ObjectUnwrap<Column> Column::shift_right_unsigned(Column const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::SHIFT_RIGHT_UNSIGNED);
}

ObjectUnwrap<Column> Column::shift_right_unsigned(Scalar const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::SHIFT_RIGHT_UNSIGNED);
}

ObjectUnwrap<Column> Column::log_base(Column const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::LOG_BASE);
}

ObjectUnwrap<Column> Column::log_base(Scalar const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::LOG_BASE);
}

ObjectUnwrap<Column> Column::atan2(Column const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::ATAN2);
}

ObjectUnwrap<Column> Column::atan2(Scalar const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::ATAN2);
}

ObjectUnwrap<Column> Column::null_equals(Column const& other) const {
  return this->binary_operation(other, cudf::binary_operator::NULL_EQUALS, cudf::type_id::BOOL8);
}

ObjectUnwrap<Column> Column::null_equals(Scalar const& other) const {
  return this->binary_operation(other, cudf::binary_operator::NULL_EQUALS, cudf::type_id::BOOL8);
}

ObjectUnwrap<Column> Column::null_max(Column const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::NULL_MAX);
}

ObjectUnwrap<Column> Column::null_max(Scalar const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::NULL_MAX);
}

ObjectUnwrap<Column> Column::null_min(Column const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::NULL_MIN);
}

ObjectUnwrap<Column> Column::null_min(Scalar const& other) const {
  return auto_binary_operation(*this, other, cudf::binary_operator::NULL_MIN);
}

// Private (JS-facing) impls

Napi::Value Column::add(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  if (Column::is_instance(rhs)) { return *this + *Column::Unwrap(rhs.ToObject()); }
  if (Scalar::is_instance(rhs)) { return *this + *Scalar::Unwrap(rhs.ToObject()); }
  if (rhs.IsBigInt()) { return *this + *Scalar::New(rhs.As<Napi::BigInt>()); }
  if (rhs.IsNumber()) { return *this + *Scalar::New(rhs.As<Napi::Number>()); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.add expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::sub(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  if (Column::is_instance(rhs)) { return *this - *Column::Unwrap(rhs.ToObject()); }
  if (Scalar::is_instance(rhs)) { return *this - *Scalar::Unwrap(rhs.ToObject()); }
  if (rhs.IsBigInt()) { return *this - *Scalar::New(rhs.As<Napi::BigInt>()); }
  if (rhs.IsNumber()) { return *this - *Scalar::New(rhs.As<Napi::Number>()); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.sub expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::mul(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  if (Column::is_instance(rhs)) { return *this * *Column::Unwrap(rhs.ToObject()); }
  if (Scalar::is_instance(rhs)) { return *this * *Scalar::Unwrap(rhs.ToObject()); }
  if (rhs.IsBigInt()) { return *this * *Scalar::New(rhs.As<Napi::BigInt>()); }
  if (rhs.IsNumber()) { return *this * *Scalar::New(rhs.As<Napi::Number>()); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.mul expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::div(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  if (Column::is_instance(rhs)) { return *this / *Column::Unwrap(rhs.ToObject()); }
  if (Scalar::is_instance(rhs)) { return *this / *Scalar::Unwrap(rhs.ToObject()); }
  if (rhs.IsBigInt()) { return *this / *Scalar::New(rhs.As<Napi::BigInt>()); }
  if (rhs.IsNumber()) { return *this / *Scalar::New(rhs.As<Napi::Number>()); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.div expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::true_div(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  if (Column::is_instance(rhs)) { return true_div(*Column::Unwrap(rhs.ToObject())); }
  if (Scalar::is_instance(rhs)) { return true_div(*Scalar::Unwrap(rhs.ToObject())); }
  if (rhs.IsBigInt()) { return true_div(*Scalar::New(rhs.As<Napi::BigInt>())); }
  if (rhs.IsNumber()) { return true_div(*Scalar::New(rhs.As<Napi::Number>())); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.true_div expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::floor_div(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  if (Column::is_instance(rhs)) { return floor_div(*Column::Unwrap(rhs.ToObject())); }
  if (Scalar::is_instance(rhs)) { return floor_div(*Scalar::Unwrap(rhs.ToObject())); }
  if (rhs.IsBigInt()) { return floor_div(*Scalar::New(rhs.As<Napi::BigInt>())); }
  if (rhs.IsNumber()) { return floor_div(*Scalar::New(rhs.As<Napi::Number>())); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.floor_div expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::mod(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  if (Column::is_instance(rhs)) { return *this % *Column::Unwrap(rhs.ToObject()); }
  if (Scalar::is_instance(rhs)) { return *this % *Scalar::Unwrap(rhs.ToObject()); }
  if (rhs.IsBigInt()) { return *this % *Scalar::New(rhs.As<Napi::BigInt>()); }
  if (rhs.IsNumber()) { return *this % *Scalar::New(rhs.As<Napi::Number>()); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.mod expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::pow(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  if (Column::is_instance(rhs)) { return pow(*Column::Unwrap(rhs.ToObject())); }
  if (Scalar::is_instance(rhs)) { return pow(*Scalar::Unwrap(rhs.ToObject())); }
  if (rhs.IsBigInt()) { return pow(*Scalar::New(rhs.As<Napi::BigInt>())); }
  if (rhs.IsNumber()) { return pow(*Scalar::New(rhs.As<Napi::Number>())); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.pow expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::eq(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  if (Column::is_instance(rhs)) { return *this == *Column::Unwrap(rhs.ToObject()); }
  if (Scalar::is_instance(rhs)) { return *this == *Scalar::Unwrap(rhs.ToObject()); }
  if (rhs.IsBigInt()) { return *this == *Scalar::New(rhs.As<Napi::BigInt>()); }
  if (rhs.IsNumber()) { return *this == *Scalar::New(rhs.As<Napi::Number>()); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.eq expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::ne(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  if (Column::is_instance(rhs)) { return *this != *Column::Unwrap(rhs.ToObject()); }
  if (Scalar::is_instance(rhs)) { return *this != *Scalar::Unwrap(rhs.ToObject()); }
  if (rhs.IsBigInt()) { return *this != *Scalar::New(rhs.As<Napi::BigInt>()); }
  if (rhs.IsNumber()) { return *this != *Scalar::New(rhs.As<Napi::Number>()); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.ne expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::lt(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  if (Column::is_instance(rhs)) { return *this < *Column::Unwrap(rhs.ToObject()); }
  if (Scalar::is_instance(rhs)) { return *this < *Scalar::Unwrap(rhs.ToObject()); }
  if (rhs.IsBigInt()) { return *this < *Scalar::New(rhs.As<Napi::BigInt>()); }
  if (rhs.IsNumber()) { return *this < *Scalar::New(rhs.As<Napi::Number>()); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.lt expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::le(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  if (Column::is_instance(rhs)) { return *this <= *Column::Unwrap(rhs.ToObject()); }
  if (Scalar::is_instance(rhs)) { return *this <= *Scalar::Unwrap(rhs.ToObject()); }
  if (rhs.IsBigInt()) { return *this <= *Scalar::New(rhs.As<Napi::BigInt>()); }
  if (rhs.IsNumber()) { return *this <= *Scalar::New(rhs.As<Napi::Number>()); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.le expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::gt(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  if (Column::is_instance(rhs)) { return *this > *Column::Unwrap(rhs.ToObject()); }
  if (Scalar::is_instance(rhs)) { return *this > *Scalar::Unwrap(rhs.ToObject()); }
  if (rhs.IsBigInt()) { return *this > *Scalar::New(rhs.As<Napi::BigInt>()); }
  if (rhs.IsNumber()) { return *this > *Scalar::New(rhs.As<Napi::Number>()); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.gt expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::ge(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  if (Column::is_instance(rhs)) { return *this >= *Column::Unwrap(rhs.ToObject()); }
  if (Scalar::is_instance(rhs)) { return *this >= *Scalar::Unwrap(rhs.ToObject()); }
  if (rhs.IsBigInt()) { return *this >= *Scalar::New(rhs.As<Napi::BigInt>()); }
  if (rhs.IsNumber()) { return *this >= *Scalar::New(rhs.As<Napi::Number>()); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.ge expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::bitwise_and(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  if (Column::is_instance(rhs)) { return *this & *Column::Unwrap(rhs.ToObject()); }
  if (Scalar::is_instance(rhs)) { return *this & *Scalar::Unwrap(rhs.ToObject()); }
  if (rhs.IsBigInt()) { return *this & *Scalar::New(rhs, type()); }
  if (rhs.IsNumber()) { return *this & *Scalar::New(rhs, type()); }
  NAPI_THROW(Napi::Error::New(info.Env(),
                              "Column.bitwise_and expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::bitwise_or(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  if (Column::is_instance(rhs)) { return *this | *Column::Unwrap(rhs.ToObject()); }
  if (Scalar::is_instance(rhs)) { return *this | *Scalar::Unwrap(rhs.ToObject()); }
  if (rhs.IsBigInt()) { return *this | *Scalar::New(rhs, type()); }
  if (rhs.IsNumber()) { return *this | *Scalar::New(rhs, type()); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.bitwise_or expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::bitwise_xor(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  if (Column::is_instance(rhs)) { return *this ^ *Column::Unwrap(rhs.ToObject()); }
  if (Scalar::is_instance(rhs)) { return *this ^ *Scalar::Unwrap(rhs.ToObject()); }
  if (rhs.IsBigInt()) { return *this ^ *Scalar::New(rhs, type()); }
  if (rhs.IsNumber()) { return *this ^ *Scalar::New(rhs, type()); }
  NAPI_THROW(Napi::Error::New(info.Env(),
                              "Column.bitwise_xor expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::logical_and(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  if (Column::is_instance(rhs)) { return *this && *Column::Unwrap(rhs.ToObject()); }
  if (Scalar::is_instance(rhs)) { return *this && *Scalar::Unwrap(rhs.ToObject()); }
  if (rhs.IsBigInt()) { return *this && *Scalar::New(rhs.As<Napi::BigInt>()); }
  if (rhs.IsNumber()) { return *this && *Scalar::New(rhs.As<Napi::Number>()); }
  NAPI_THROW(Napi::Error::New(info.Env(),
                              "Column.logical_and expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::logical_or(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  if (Column::is_instance(rhs)) { return *this || *Column::Unwrap(rhs.ToObject()); }
  if (Scalar::is_instance(rhs)) { return *this || *Scalar::Unwrap(rhs.ToObject()); }
  if (rhs.IsBigInt()) { return *this || *Scalar::New(rhs.As<Napi::BigInt>()); }
  if (rhs.IsNumber()) { return *this || *Scalar::New(rhs.As<Napi::Number>()); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.logical_or expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::coalesce(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  if (Column::is_instance(rhs)) { return coalesce(*Column::Unwrap(rhs.ToObject())); }
  if (Scalar::is_instance(rhs)) { return coalesce(*Scalar::Unwrap(rhs.ToObject())); }
  if (rhs.IsBigInt()) { return coalesce(*Scalar::New(rhs.As<Napi::BigInt>())); }
  if (rhs.IsNumber()) { return coalesce(*Scalar::New(rhs.As<Napi::Number>())); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.coalesce expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::shift_left(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  if (Column::is_instance(rhs)) { return *this << *Column::Unwrap(rhs.ToObject()); }
  if (Scalar::is_instance(rhs)) { return *this << *Scalar::Unwrap(rhs.ToObject()); }
  if (rhs.IsBigInt()) { return *this << *Scalar::New(rhs, type()); }
  if (rhs.IsNumber()) { return *this << *Scalar::New(rhs, type()); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.shift_left expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::shift_right(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  if (Column::is_instance(rhs)) { return *this >> *Column::Unwrap(rhs.ToObject()); }
  if (Scalar::is_instance(rhs)) { return *this >> *Scalar::Unwrap(rhs.ToObject()); }
  if (rhs.IsBigInt()) { return *this >> *Scalar::New(rhs, type()); }
  if (rhs.IsNumber()) { return *this >> *Scalar::New(rhs, type()); }
  NAPI_THROW(Napi::Error::New(info.Env(),
                              "Column.shift_right expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::shift_right_unsigned(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  if (Column::is_instance(rhs)) { return shift_right_unsigned(*Column::Unwrap(rhs.ToObject())); }
  if (Scalar::is_instance(rhs)) { return shift_right_unsigned(*Scalar::Unwrap(rhs.ToObject())); }
  if (rhs.IsBigInt()) { return shift_right_unsigned(*Scalar::New(rhs, type())); }
  if (rhs.IsNumber()) { return shift_right_unsigned(*Scalar::New(rhs, type())); }
  NAPI_THROW(Napi::Error::New(
    info.Env(), "Column.shift_right_unsigned expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::log_base(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  if (Column::is_instance(rhs)) { return log_base(*Column::Unwrap(rhs.ToObject())); }
  if (Scalar::is_instance(rhs)) { return log_base(*Scalar::Unwrap(rhs.ToObject())); }
  if (rhs.IsBigInt()) { return log_base(*Scalar::New(rhs.As<Napi::BigInt>())); }
  if (rhs.IsNumber()) { return log_base(*Scalar::New(rhs.As<Napi::Number>())); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.log_base expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::atan2(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  if (Column::is_instance(rhs)) { return atan2(*Column::Unwrap(rhs.ToObject())); }
  if (Scalar::is_instance(rhs)) { return atan2(*Scalar::Unwrap(rhs.ToObject())); }
  if (rhs.IsBigInt()) { return atan2(*Scalar::New(rhs.As<Napi::BigInt>())); }
  if (rhs.IsNumber()) { return atan2(*Scalar::New(rhs.As<Napi::Number>())); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.atan2 expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::null_equals(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  if (Column::is_instance(rhs)) { return null_equals(*Column::Unwrap(rhs.ToObject())); }
  if (Scalar::is_instance(rhs)) { return null_equals(*Scalar::Unwrap(rhs.ToObject())); }
  if (rhs.IsBigInt()) { return null_equals(*Scalar::New(rhs.As<Napi::BigInt>())); }
  if (rhs.IsNumber()) { return null_equals(*Scalar::New(rhs.As<Napi::Number>())); }
  NAPI_THROW(Napi::Error::New(info.Env(),
                              "Column.null_equals expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::null_max(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  if (Column::is_instance(rhs)) { return null_max(*Column::Unwrap(rhs.ToObject())); }
  if (Scalar::is_instance(rhs)) { return null_max(*Scalar::Unwrap(rhs.ToObject())); }
  if (rhs.IsBigInt()) { return null_max(*Scalar::New(rhs.As<Napi::BigInt>())); }
  if (rhs.IsNumber()) { return null_max(*Scalar::New(rhs.As<Napi::Number>())); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.null_max expects a Column, Scalar, bigint, or number."));
}

Napi::Value Column::null_min(Napi::CallbackInfo const& info) {
  auto rhs = info[0];
  if (Column::is_instance(rhs)) { return null_min(*Column::Unwrap(rhs.ToObject())); }
  if (Scalar::is_instance(rhs)) { return null_min(*Scalar::Unwrap(rhs.ToObject())); }
  if (rhs.IsBigInt()) { return null_min(*Scalar::New(rhs.As<Napi::BigInt>())); }
  if (rhs.IsNumber()) { return null_min(*Scalar::New(rhs.As<Napi::Number>())); }
  NAPI_THROW(
    Napi::Error::New(info.Env(), "Column.null_min expects a Column, Scalar, bigint, or number."));
}

}  // namespace nv
