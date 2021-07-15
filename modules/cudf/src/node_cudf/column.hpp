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

#pragma once

#include <node_cudf/scalar.hpp>
#include <node_cudf/utilities/dtypes.hpp>

#include <node_rmm/device_buffer.hpp>

#include <nv_node/objectwrap.hpp>
#include <nv_node/utilities/args.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/replace.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>

#include <rmm/device_buffer.hpp>
#include "cudf/reduction.hpp"

#include <napi.h>

namespace nv {

/**
 * @brief An owning wrapper around a cudf::Column.
 *
 */
struct Column : public EnvLocalObjectWrap<Column> {
  /**
   * @brief Initialize and export the Column JavaScript constructor and prototype.
   *
   * @param env The active JavaScript environment.
   * @param exports The exports object to decorate.
   * @return Napi::Function The Column constructor function.
   */
  static Napi::Function Init(Napi::Env const& env, Napi::Object exports);

  /**
   * @brief Construct a new Column instance from a cudf::column.
   *
   * @param column The column in device memory.
   */
  static wrapper_t New(Napi::Env const& env, std::unique_ptr<cudf::column> column);

  /**
   * @brief Construct a new Column instance from JavaScript.
   *
   */
  Column(CallbackArgs const& args);

  /**
   * @brief Returns the column's logical element type
   */
  inline cudf::data_type type() const noexcept { return arrow_to_cudf_type(type_.Value()); }

  /**
   * @brief Returns the number of elements
   */
  inline cudf::size_type size() const noexcept { return size_; }

  /**
   * @brief Returns the data element offset
   */
  inline cudf::size_type offset() const noexcept { return offset_; }

  /**
   * @brief Return a const pointer to the data buffer
   */
  inline DeviceBuffer::wrapper_t data() const { return data_.Value(); }

  /**
   * @brief Return a const pointer to the null bitmask buffer
   */
  inline DeviceBuffer::wrapper_t null_mask() const { return null_mask_.Value(); }

  /**
   * @brief Sets the column's null value indicator bitmask to `new_null_mask`.
   *
   * @throw cudf::logic_error if new_null_count is larger than 0 and the size
   * of `new_null_mask` does not match the size of this column.
   *
   * @param new_null_mask New null value indicator bitmask (lvalue overload &
   * copied) to set the column's null value indicator mask. May be empty if
   * `new_null_count` is 0 or `UNKOWN_NULL_COUNT`.
   * @param new_null_count Optional, the count of null elements. If unknown,
   * specify `UNKNOWN_NULL_COUNT` to indicate that the null count should be
   * computed on the first invocation of `null_count()`.
   */
  void set_null_mask(Napi::Value const& new_null_mask,
                     cudf::size_type new_null_count = cudf::UNKNOWN_NULL_COUNT);

  /**
   * @brief Updates the count of null elements.
   *
   * @note `UNKNOWN_NULL_COUNT` can be specified as `new_null_count` to force
   * the next invocation of `null_count()` to recompute the null count from the
   * null mask.
   *
   * @throw cudf::logic_error if `new_null_count > 0 and nullable() == false`
   *
   * @param new_null_count The new null count.
   */
  void set_null_count(cudf::size_type new_null_count);

  /**
   * @brief Indicates whether it is possible for the column to contain null
   * values, i.e., it has an allocated null mask.
   *
   * This may return `false` iff `null_count() == 0`.
   *
   * May return true even if `null_count() == 0`. This function simply indicates
   * whether the column has an allocated null mask.
   *
   * @return true The column can hold null values
   * @return false The column cannot hold null values
   */
  inline bool nullable() const { return null_mask()->size() > 0; }

  /**
   * @brief Returns the count of null elements.
   *
   * @note If the column was constructed with `UNKNOWN_NULL_COUNT`, or if at any
   * point `set_null_count(UNKNOWN_NULL_COUNT)` was invoked, then the
   * first invocation of `null_count()` will compute and store the count of null
   * elements indicated by the `null_mask` (if it exists).
   */
  cudf::size_type null_count() const;

  /**
   * @brief Creates an immutable, non-owning view of the column's data and
   * children.
   *
   * @return cudf::column_view The immutable, non-owning view
   */
  cudf::column_view view() const;

  /**
   * @brief Returns the number of child columns
   */
  cudf::size_type num_children() const { return children_.Value().Length(); }

  /**
   * @brief Returns a const reference to the specified child
   *
   * @param child_index Index of the desired child
   * @return column const& Const reference to the desired child
   */
  Column::wrapper_t child(cudf::size_type child_index) const noexcept {
    return children_.Value().Get(child_index).ToObject();
  };

  /**
   * @brief Creates a mutable, non-owning view of the column's data and
   * children.
   *
   * @note Creating a mutable view of a `column` invalidates the `column`'s
   * `null_count()` by setting it to `UNKNOWN_NULL_COUNT`. The user can
   * either explicitly update the null count with `set_null_count()`, or
   * if not, the null count will be recomputed on the next invocation of
   *`null_count()`.
   *
   * @return cudf::mutable_column_view The mutable, non-owning view
   */
  cudf::mutable_column_view mutable_view();

  /**
   * @brief Implicit conversion operator to a `column_view`.
   *
   * This allows passing a `column` object directly into a function that
   * requires a `column_view`. The conversion is automatic.
   *
   * @return cudf::column_view Immutable, non-owning `column_view`
   */
  operator cudf::column_view() const { return this->view(); };

  /**
   * @brief Implicit conversion operator to a `mutable_column_view`.
   *
   * This allows pasing a `column` object into a function that accepts a
   *`mutable_column_view`. The conversion is automatic.

   * @note Creating a mutable view of a `column` invalidates the `column`'s
   * `null_count()` by setting it to `UNKNOWN_NULL_COUNT`. For best performance,
   * the user should explicitly update the null count with `set_null_count()`.
   * Otherwise, the null count will be recomputed on the next invocation of
   * `null_count()`.
   *
   * @return cudf::mutable_column_view Mutable, non-owning `mutable_column_view`
   */
  operator cudf::mutable_column_view() { return this->mutable_view(); };

  // column/reductions.cpp

  /**
   * @copydoc cudf::minmax(cudf::column_view const& col, rmm::mr::device_memory_resource* mr)
   *
   * @return std::pair<Scalar, Scalar>
   */
  std::pair<Scalar::wrapper_t, Scalar::wrapper_t> minmax(
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  /**
   * @copydoc cudf::reduce(cudf::column_view const &col, std::unique_ptr<aggregation> const &agg,
   * data_type output_dtype, rmm::mr::device_memory_resource* mr)
   *
   * @return Scalar
   */
  Scalar::wrapper_t reduce(
    std::unique_ptr<cudf::aggregation> const& agg,
    cudf::data_type const& output_dtype,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  /**
   * @copydoc cudf::scan(cudf::column_view const &col, std::unique_ptr<aggregation> const &agg,
   * cudf::scan_type inclusive, cudf::null_policy null_handling, rmm::mr::device_memory_resource*
   * mr)
   *
   * @return Column
   */
  Column::wrapper_t scan(
    std::unique_ptr<cudf::aggregation> const& agg,
    cudf::scan_type inclusive,
    cudf::null_policy null_handling     = cudf::null_policy::EXCLUDE,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  /**
   * @copydoc cudf::sum(rmm::mr::device_memory_resource* mr)
   *
   * @return Scalar
   */
  Scalar::wrapper_t sum(
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  /**
   * @copydoc cudf::product(rmm::mr::device_memory_resource* mr)
   *
   * @return Scalar
   */
  Scalar::wrapper_t product(
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  /**
   * @copydoc cudf::sum_of_squares(rmm::mr::device_memory_resource* mr)
   *
   * @return Scalar
   */
  Scalar::wrapper_t sum_of_squares(
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  /**
   * @brief Returns a reduce computation based on cudf::make_mean_aggregation()
   *
   * @param rmm::mr::device_memory_resource* mr
   *
   * @return Scalar
   */
  Scalar::wrapper_t mean(
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  /**
   * @brief Returns a reduce computation based on cudf::make_median_aggregation()
   *
   * @param rmm::mr::device_memory_resource* mr
   *
   * @return Scalar
   */
  Scalar::wrapper_t median(
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  /**
   * @brief Returns a reduce computation based on cudf::make_nunique_aggregation()
   *
   * @param bool dropna if true, compute using cudf::null_policy::EXCLUDE, else use
   * cudf::null_policy::INCLUDE
   * @param rmm::mr::device_memory_resource* mr
   * @return Scalar
   */
  Scalar::wrapper_t nunique(
    bool dropna                         = true,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  /**
   * @brief Returns a reduce computation based on cudf::make_variance_aggregation(ddof)
   *
   * @param cudf::size_type ddof Delta Degrees of Freedom. The divisor used in calculations is
   * N - ddof, where N represents the number of elements. default is 1
   * @param rmm::mr::device_memory_resource* mr
   * @return Scalar
   */
  Scalar::wrapper_t variance(
    cudf::size_type ddof                = 1,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  /**
   * @brief Returns a reduce computation based on cudf::make_std_aggregation(ddof)
   *
   * @param cudf::size_type ddof Delta Degrees of Freedom. The divisor used in calculations is
   * N - ddof, where N represents the number of elements. default is 1
   * @param rmm::mr::device_memory_resource* mr
   * @return Scalar
   */
  Scalar::wrapper_t std(
    cudf::size_type ddof                = 1,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  /**
   * @brief Returns a reduce computation based on cudf::make_quantile_aggregation(q, i)
   *
   * @param double q  the quantile(s) to compute, 0<=q<=1, default 0.5
   * @param rmm::mr::device_memory_resource* mr
   * @return Scalar
   */
  Scalar::wrapper_t quantile(
    double q                            = 0.5,
    cudf::interpolation i               = cudf::interpolation::LINEAR,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  /**
   * @brief Return the cumulative max
   *
   * @return Scalar
   */
  Column::wrapper_t cumulative_max(
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  /**
   * @brief Return the cumulative minimum
   *
   * @return Scalar
   */
  Column::wrapper_t cumulative_min(
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  /**
   * @brief Return the cumulative product
   *
   * @return Scalar
   */
  Column::wrapper_t cumulative_product(
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  /**
   * @brief Return the cumulative sum
   *
   * @return Scalar
   */
  Column::wrapper_t cumulative_sum(
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  // column/binaryop.cpp

  // cudf::binary_operator::ADD
  Column::wrapper_t operator+(Column const& other) const;
  Column::wrapper_t operator+(Scalar const& other) const;
  Column::wrapper_t add(
    Column const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  Column::wrapper_t add(
    Scalar const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  // cudf::binary_operator::SUB
  Column::wrapper_t operator-(Column const& other) const;
  Column::wrapper_t operator-(Scalar const& other) const;
  Column::wrapper_t sub(
    Column const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  Column::wrapper_t sub(
    Scalar const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  // cudf::binary_operator::MUL
  Column::wrapper_t operator*(Column const& other) const;
  Column::wrapper_t operator*(Scalar const& other) const;
  Column::wrapper_t mul(
    Column const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  Column::wrapper_t mul(
    Scalar const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  // cudf::binary_operator::DIV
  Column::wrapper_t operator/(Column const& other) const;
  Column::wrapper_t operator/(Scalar const& other) const;
  Column::wrapper_t div(
    Column const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  Column::wrapper_t div(
    Scalar const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  // cudf::binary_operator::TRUE_DIV
  Column::wrapper_t true_div(
    Column const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  Column::wrapper_t true_div(
    Scalar const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  // cudf::binary_operator::FLOOR_DIV
  Column::wrapper_t floor_div(
    Column const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  Column::wrapper_t floor_div(
    Scalar const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  // cudf::binary_operator::MOD
  Column::wrapper_t operator%(Column const& other) const;
  Column::wrapper_t operator%(Scalar const& other) const;
  Column::wrapper_t mod(
    Column const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  Column::wrapper_t mod(
    Scalar const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  // cudf::binary_operator::POW
  Column::wrapper_t pow(
    Column const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  Column::wrapper_t pow(
    Scalar const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  // cudf::binary_operator::EQUAL
  Column::wrapper_t operator==(Column const& other) const;
  Column::wrapper_t operator==(Scalar const& other) const;
  Column::wrapper_t eq(
    Column const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  Column::wrapper_t eq(
    Scalar const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  // cudf::binary_operator::NOT_EQUAL
  Column::wrapper_t operator!=(Column const& other) const;
  Column::wrapper_t operator!=(Scalar const& other) const;
  Column::wrapper_t ne(
    Column const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  Column::wrapper_t ne(
    Scalar const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  // cudf::binary_operator::LESS
  Column::wrapper_t operator<(Column const& other) const;
  Column::wrapper_t operator<(Scalar const& other) const;
  Column::wrapper_t lt(
    Column const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  Column::wrapper_t lt(
    Scalar const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  // cudf::binary_operator::GREATER
  Column::wrapper_t operator>(Column const& other) const;
  Column::wrapper_t operator>(Scalar const& other) const;
  Column::wrapper_t gt(
    Column const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  Column::wrapper_t gt(
    Scalar const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  // cudf::binary_operator::LESS_EQUAL
  Column::wrapper_t operator<=(Column const& other) const;
  Column::wrapper_t operator<=(Scalar const& other) const;
  Column::wrapper_t le(
    Column const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  Column::wrapper_t le(
    Scalar const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  // cudf::binary_operator::GREATER_EQUAL
  Column::wrapper_t operator>=(Column const& other) const;
  Column::wrapper_t operator>=(Scalar const& other) const;
  Column::wrapper_t ge(
    Column const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  Column::wrapper_t ge(
    Scalar const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  // cudf::binary_operator::BITWISE_AND
  Column::wrapper_t operator&(Column const& other) const;
  Column::wrapper_t operator&(Scalar const& other) const;
  Column::wrapper_t bitwise_and(
    Column const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  Column::wrapper_t bitwise_and(
    Scalar const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  // cudf::binary_operator::BITWISE_OR
  Column::wrapper_t operator|(Column const& other) const;
  Column::wrapper_t operator|(Scalar const& other) const;
  Column::wrapper_t bitwise_or(
    Column const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  Column::wrapper_t bitwise_or(
    Scalar const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  // cudf::binary_operator::BITWISE_XOR
  Column::wrapper_t operator^(Column const& other) const;
  Column::wrapper_t operator^(Scalar const& other) const;
  Column::wrapper_t bitwise_xor(
    Column const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  Column::wrapper_t bitwise_xor(
    Scalar const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  // cudf::binary_operator::LOGICAL_AND
  Column::wrapper_t operator&&(Column const& other) const;
  Column::wrapper_t operator&&(Scalar const& other) const;
  Column::wrapper_t logical_and(
    Column const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  Column::wrapper_t logical_and(
    Scalar const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  // cudf::binary_operator::LOGICAL_OR
  Column::wrapper_t operator||(Column const& other) const;
  Column::wrapper_t operator||(Scalar const& other) const;
  Column::wrapper_t logical_or(
    Column const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  Column::wrapper_t logical_or(
    Scalar const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  // cudf::binary_operator::COALESCE
  Column::wrapper_t coalesce(
    Column const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  Column::wrapper_t coalesce(
    Scalar const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  // cudf::binary_operator::SHIFT_LEFT
  Column::wrapper_t operator<<(Column const& other) const;
  Column::wrapper_t operator<<(Scalar const& other) const;
  Column::wrapper_t shift_left(
    Column const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  Column::wrapper_t shift_left(
    Scalar const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  // cudf::binary_operator::SHIFT_RIGHT
  Column::wrapper_t operator>>(Column const& other) const;
  Column::wrapper_t operator>>(Scalar const& other) const;
  Column::wrapper_t shift_right(
    Column const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  Column::wrapper_t shift_right(
    Scalar const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  // cudf::binary_operator::SHIFT_RIGHT_UNSIGNED
  Column::wrapper_t shift_right_unsigned(
    Column const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  Column::wrapper_t shift_right_unsigned(
    Scalar const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  // cudf::binary_operator::LOG_BASE
  Column::wrapper_t log_base(
    Column const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  Column::wrapper_t log_base(
    Scalar const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  // cudf::binary_operator::ATAN2
  Column::wrapper_t atan2(
    Column const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  Column::wrapper_t atan2(
    Scalar const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  // cudf::binary_operator::NULL_EQUALS
  Column::wrapper_t null_equals(
    Column const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  Column::wrapper_t null_equals(
    Scalar const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  // cudf::binary_operator::NULL_MAX
  Column::wrapper_t null_max(
    Column const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  Column::wrapper_t null_max(
    Scalar const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  // cudf::binary_operator::NULL_MIN
  Column::wrapper_t null_min(
    Column const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
  Column::wrapper_t null_min(
    Scalar const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  Column::wrapper_t operator[](Column const& gather_map) const;

  Column::wrapper_t binary_operation(
    Column const& rhs,
    cudf::binary_operator op,
    cudf::type_id output_type,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  Column::wrapper_t binary_operation(
    Scalar const& rhs,
    cudf::binary_operator op,
    cudf::type_id output_type,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  // column/concatenate.cpp
  Column::wrapper_t concat(
    cudf::column_view const& other,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  // column/stream_compaction.cpp
  Column::wrapper_t apply_boolean_mask(
    Column const& boolean_mask,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  Column::wrapper_t drop_nulls(
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  Column::wrapper_t drop_nans(
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  // column/filling.cpp
  static Column::wrapper_t sequence(
    Napi::Env const& env,
    cudf::size_type size,
    cudf::scalar const& init,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  static Column::wrapper_t sequence(
    Napi::Env const& env,
    cudf::size_type size,
    cudf::scalar const& init,
    cudf::scalar const& step,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  // column/transform.cpp
  std::pair<std::unique_ptr<rmm::device_buffer>, cudf::size_type> nans_to_nulls(
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  // column/copying.cpp
  Column::wrapper_t gather(
    Column const& gather_map,
    cudf::out_of_bounds_policy bounds_policy = cudf::out_of_bounds_policy::DONT_CHECK,
    rmm::mr::device_memory_resource* mr      = rmm::mr::get_current_device_resource()) const;

  // column/filling.cpp
  Column::wrapper_t fill(
    cudf::size_type begin,
    cudf::size_type end,
    cudf::scalar const& value,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  // column/strings/json.cpp
  Column::wrapper_t get_json_object(
    std::string const& json_path,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  // column/replace.cpp
  Column::wrapper_t replace_nulls(
    cudf::column_view const& replacement,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  Column::wrapper_t replace_nulls(
    cudf::scalar const& replacement,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  Column::wrapper_t replace_nulls(
    cudf::replace_policy const& replace_policy,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  Column::wrapper_t replace_nans(
    cudf::column_view const& replacement,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  Column::wrapper_t replace_nans(
    cudf::scalar const& replacement,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  // column/unaryop.cpp
  Column::wrapper_t cast(
    cudf::data_type out_type,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  Column::wrapper_t is_null(
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  Column::wrapper_t is_valid(
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  Column::wrapper_t is_nan(
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  Column::wrapper_t is_not_nan(
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  Column::wrapper_t unary_operation(
    cudf::unary_operator op,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  // column/re.cpp
  Column::wrapper_t contains_re(
    std::string const& pattern,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  Column::wrapper_t count_re(
    std::string const& pattern,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  // TODO: findall_re

  Column::wrapper_t matches_re(
    std::string const& pattern,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  // column/convert.cpp
  Column::wrapper_t string_is_float(
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  Column::wrapper_t strings_from_floats(
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  Column::wrapper_t strings_to_floats(
    cudf::data_type out_type,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  Column::wrapper_t string_is_integer(
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  Column::wrapper_t strings_from_integers(
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  Column::wrapper_t strings_to_integers(
    cudf::data_type out_type,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

 private:
  cudf::size_type size_{};                         ///< The number of elements in the column
  cudf::size_type offset_{};                       ///< The offset of elements in the data
  Napi::Reference<Napi::Object> type_;             ///< Logical type of elements in the column
  Napi::Reference<DeviceBuffer::wrapper_t> data_;  ///< Dense, contiguous, type erased device memory
                                                   ///< buffer containing the column elements
  Napi::Reference<DeviceBuffer::wrapper_t> null_mask_;  ///< Bitmask used to represent null values.
                                                        ///< May be empty if `null_count() == 0`
  mutable cudf::size_type null_count_{cudf::UNKNOWN_NULL_COUNT};  ///< The number of null elements
  Napi::Reference<Napi::Array> children_;  ///< Depending on element type, child
                                           ///< columns may contain additional data

  Napi::Value type(Napi::CallbackInfo const& info);
  void type(Napi::CallbackInfo const& info, Napi::Value const& value);

  Napi::Value offset(Napi::CallbackInfo const& info);
  Napi::Value size(Napi::CallbackInfo const& info);
  Napi::Value data(Napi::CallbackInfo const& info);
  Napi::Value null_mask(Napi::CallbackInfo const& info);
  Napi::Value has_nulls(Napi::CallbackInfo const& info);
  Napi::Value null_count(Napi::CallbackInfo const& info);
  Napi::Value is_nullable(Napi::CallbackInfo const& info);
  Napi::Value num_children(Napi::CallbackInfo const& info);

  Napi::Value gather(Napi::CallbackInfo const& info);
  Napi::Value copy(Napi::CallbackInfo const& info);

  Napi::Value get_child(Napi::CallbackInfo const& info);

  Napi::Value get_value(Napi::CallbackInfo const& info);

  void set_null_mask(Napi::CallbackInfo const& info);
  void set_null_count(Napi::CallbackInfo const& info);

  // column/binaryop.cpp
  Napi::Value add(Napi::CallbackInfo const& info);
  Napi::Value sub(Napi::CallbackInfo const& info);
  Napi::Value mul(Napi::CallbackInfo const& info);
  Napi::Value div(Napi::CallbackInfo const& info);
  Napi::Value true_div(Napi::CallbackInfo const& info);
  Napi::Value floor_div(Napi::CallbackInfo const& info);
  Napi::Value mod(Napi::CallbackInfo const& info);
  Napi::Value pow(Napi::CallbackInfo const& info);
  Napi::Value eq(Napi::CallbackInfo const& info);
  Napi::Value ne(Napi::CallbackInfo const& info);
  Napi::Value lt(Napi::CallbackInfo const& info);
  Napi::Value gt(Napi::CallbackInfo const& info);
  Napi::Value le(Napi::CallbackInfo const& info);
  Napi::Value ge(Napi::CallbackInfo const& info);
  Napi::Value bitwise_and(Napi::CallbackInfo const& info);
  Napi::Value bitwise_or(Napi::CallbackInfo const& info);
  Napi::Value bitwise_xor(Napi::CallbackInfo const& info);
  Napi::Value logical_and(Napi::CallbackInfo const& info);
  Napi::Value logical_or(Napi::CallbackInfo const& info);
  Napi::Value coalesce(Napi::CallbackInfo const& info);
  Napi::Value shift_left(Napi::CallbackInfo const& info);
  Napi::Value shift_right(Napi::CallbackInfo const& info);
  Napi::Value shift_right_unsigned(Napi::CallbackInfo const& info);
  Napi::Value log_base(Napi::CallbackInfo const& info);
  Napi::Value atan2(Napi::CallbackInfo const& info);
  Napi::Value null_equals(Napi::CallbackInfo const& info);
  Napi::Value null_max(Napi::CallbackInfo const& info);
  Napi::Value null_min(Napi::CallbackInfo const& info);

  // column/concatenate.cpp
  Napi::Value concat(Napi::CallbackInfo const& info);

  // column/filling.cpp
  Napi::Value fill(Napi::CallbackInfo const& info);
  void fill_in_place(Napi::CallbackInfo const& info);

  // column/stream_compaction.cpp
  Napi::Value drop_nulls(Napi::CallbackInfo const& info);
  Napi::Value drop_nans(Napi::CallbackInfo const& info);

  // column/filling.cpp
  static Napi::Value sequence(Napi::CallbackInfo const& info);

  // column/transform.cpp
  Napi::Value nans_to_nulls(Napi::CallbackInfo const& info);

  // column/reductions.cpp
  Napi::Value min(Napi::CallbackInfo const& info);
  Napi::Value max(Napi::CallbackInfo const& info);
  Napi::Value minmax(Napi::CallbackInfo const& info);
  Napi::Value sum(Napi::CallbackInfo const& info);
  Napi::Value product(Napi::CallbackInfo const& info);
  Napi::Value any(Napi::CallbackInfo const& info);
  Napi::Value all(Napi::CallbackInfo const& info);
  Napi::Value sum_of_squares(Napi::CallbackInfo const& info);
  Napi::Value mean(Napi::CallbackInfo const& info);
  Napi::Value median(Napi::CallbackInfo const& info);
  Napi::Value nunique(Napi::CallbackInfo const& info);
  Napi::Value variance(Napi::CallbackInfo const& info);
  Napi::Value std(Napi::CallbackInfo const& info);
  Napi::Value quantile(Napi::CallbackInfo const& info);
  Napi::Value cumulative_max(Napi::CallbackInfo const& info);
  Napi::Value cumulative_min(Napi::CallbackInfo const& info);
  Napi::Value cumulative_product(Napi::CallbackInfo const& info);
  Napi::Value cumulative_sum(Napi::CallbackInfo const& info);

  // column/strings/json.cpp
  Napi::Value get_json_object(Napi::CallbackInfo const& info);

  // column/replace.cpp
  Napi::Value replace_nulls(Napi::CallbackInfo const& info);
  Napi::Value replace_nans(Napi::CallbackInfo const& info);

  // column/unaryop.cpp
  Napi::Value cast(Napi::CallbackInfo const& info);
  Napi::Value is_null(Napi::CallbackInfo const& info);
  Napi::Value is_valid(Napi::CallbackInfo const& info);
  Napi::Value is_nan(Napi::CallbackInfo const& info);
  Napi::Value is_not_nan(Napi::CallbackInfo const& info);
  Napi::Value sin(Napi::CallbackInfo const& info);
  Napi::Value cos(Napi::CallbackInfo const& info);
  Napi::Value tan(Napi::CallbackInfo const& info);
  Napi::Value arcsin(Napi::CallbackInfo const& info);
  Napi::Value arccos(Napi::CallbackInfo const& info);
  Napi::Value arctan(Napi::CallbackInfo const& info);
  Napi::Value sinh(Napi::CallbackInfo const& info);
  Napi::Value cosh(Napi::CallbackInfo const& info);
  Napi::Value tanh(Napi::CallbackInfo const& info);
  Napi::Value arcsinh(Napi::CallbackInfo const& info);
  Napi::Value arccosh(Napi::CallbackInfo const& info);
  Napi::Value arctanh(Napi::CallbackInfo const& info);
  Napi::Value exp(Napi::CallbackInfo const& info);
  Napi::Value log(Napi::CallbackInfo const& info);
  Napi::Value sqrt(Napi::CallbackInfo const& info);
  Napi::Value cbrt(Napi::CallbackInfo const& info);
  Napi::Value ceil(Napi::CallbackInfo const& info);
  Napi::Value floor(Napi::CallbackInfo const& info);
  Napi::Value abs(Napi::CallbackInfo const& info);
  Napi::Value rint(Napi::CallbackInfo const& info);
  Napi::Value bit_invert(Napi::CallbackInfo const& info);
  Napi::Value unary_not(Napi::CallbackInfo const& info);

  // column/re.cpp
  Napi::Value contains_re(Napi::CallbackInfo const& info);
  Napi::Value count_re(Napi::CallbackInfo const& info);
  // Napi::Value findall_re(Napi::CallbackInfo const& info);
  Napi::Value matches_re(Napi::CallbackInfo const& info);

  // column/convert.hpp
  Napi::Value string_is_float(Napi::CallbackInfo const& info);
  Napi::Value strings_from_floats(Napi::CallbackInfo const& info);
  Napi::Value strings_to_floats(Napi::CallbackInfo const& info);
  Napi::Value string_is_integer(Napi::CallbackInfo const& info);
  Napi::Value strings_from_integers(Napi::CallbackInfo const& info);
  Napi::Value strings_to_integers(Napi::CallbackInfo const& info);
};

}  // namespace nv
