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

#include <node_cudf/column.hpp>

#include <cudf/binaryop.hpp>

namespace nv {

ObjectUnwrap<Column> Column::operator==(Column const& other) const {
  return this->binary_operation(other, cudf::binary_operator::EQUAL, cudf::type_id::BOOL8);
}

ObjectUnwrap<Column> Column::operator==(Scalar const& other) const {
  return this->binary_operation(other, cudf::binary_operator::EQUAL, cudf::type_id::BOOL8);
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

ObjectUnwrap<Column> Column::binary_operation(Column const& rhs,
                                              cudf::binary_operator op,
                                              cudf::type_id output_type) const {
  return Column::New(
    std::move(cudf::binary_operation(*this, rhs, op, cudf::data_type{output_type})));
}

ObjectUnwrap<Column> Column::binary_operation(Scalar const& rhs,
                                              cudf::binary_operator op,
                                              cudf::type_id output_type) const {
  return Column::New(
    std::move(cudf::binary_operation(*this, rhs, op, cudf::data_type{output_type})));
}

}  // namespace nv
