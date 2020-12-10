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

#include "node_cudf/scalar.hpp"
#include "node_cudf/types.hpp"

#include <node_rmm/device_buffer.hpp>
#include <nv_node/utilities/args.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <napi.h>

namespace nv {

/**
 * @brief An owning wrapper around a cudf::Column.
 *
 */
class Column : public Napi::ObjectWrap<Column> {
 public:
  /**
   * @brief Initialize and export the Column JavaScript constructor and prototype.
   *
   * @param env The active JavaScript environment.
   * @param exports The exports object to decorate.
   * @return Napi::Object The decorated exports object.
   */
  static Napi::Object Init(Napi::Env env, Napi::Object exports);

  /**
   * @brief Construct a new Column instance from JavaScript.
   *
   */
  Column(CallbackArgs const& args);

  /**
   * @brief Initialize the Column instance created by either C++ or JavaScript.
   *
   * @param data The column's data.
   * @param size The number of elements in the column.
   * @param type The element data type.
   */
  void Initialize(Napi::Object const& data, cudf::size_type size, Napi::Object const& type) {
    Initialize(data, size, type, DeviceBuffer::New(nullptr, 0), 0, 0, Napi::Array::New(Env(), 0));
  }

  /**
   * @brief Initialize the Column instance created by either C++ or JavaScript.
   *
   * @param data The column's data.
   * @param size The number of elements in the column.
   * @param type The element data type.
   * @param null_mask Optional, column's null value indicator bitmask.
   * May be empty if null_count is 0 or UNKNOWN_NULL_COUNT.
   * @param offset The element offset from the start of the underlying data.
   * @param null_count Optional, the count of null elements. If unknown, specify
   * UNKNOWN_NULL_COUNT to indicate that the null count should be computed on the first invocation
   * of `null_count()`.
   * @param children Optional Array of child columns
   */
  void Initialize(Napi::Object const& data,
                  cudf::size_type size,
                  Napi::Object const& type,
                  Napi::Object const& null_mask,
                  cudf::size_type offset,
                  cudf::size_type null_count  = cudf::UNKNOWN_NULL_COUNT,
                  Napi::Array const& children = {});

  /**
   * @brief Destructor called when the JavaScript VM garbage collects this Column
   * instance.
   *
   * @param env The active JavaScript environment.
   */
  void Finalize(Napi::Env env) override;

  /**
   * @brief Returns the column's logical element type
   */
  cudf::data_type type() const noexcept { return *DataType::Unwrap(type_.Value()); }

  /**
   * @brief Returns the number of elements
   */
  cudf::size_type size() const noexcept { return size_; }

  /**
   * @brief Return a const reference to the data buffer
   */
  DeviceBuffer const& data() const { return *DeviceBuffer::Unwrap(data_.Value()); }

  /**
   * @brief Return a const reference to the null bitmask buffer
   */
  DeviceBuffer const& null_mask() const { return *DeviceBuffer::Unwrap(null_mask_.Value()); }

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
  bool nullable() const noexcept { return (DeviceBuffer::Unwrap(null_mask_.Value())->size() > 0); }

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

 private:
  static Napi::FunctionReference constructor;

  Napi::ObjectReference type_{};       ///< Logical type of elements in the column
  cudf::size_type size_{};             ///< The number of elements in the column
  cudf::size_type offset_{};           ///< The number of elements in the column
  Napi::ObjectReference data_{};       ///< Dense, contiguous, type erased device memory
                                       ///< buffer containing the column elements
  Napi::ObjectReference null_mask_{};  ///< Bitmask used to represent null values.
                                       ///< May be empty if `null_count() == 0`
  mutable cudf::size_type null_count_{cudf::UNKNOWN_NULL_COUNT};  ///< The number of null elements
  Napi::Reference<Napi::Array> children_{};  ///< Depending on element type, child
                                             ///< columns may contain additional data

  Napi::Value type(Napi::CallbackInfo const& info);
  Napi::Value size(Napi::CallbackInfo const& info);
  Napi::Value data(Napi::CallbackInfo const& info);
  Napi::Value null_mask(Napi::CallbackInfo const& info);
  Napi::Value has_nulls(Napi::CallbackInfo const& info);
  Napi::Value null_count(Napi::CallbackInfo const& info);
  Napi::Value is_nullable(Napi::CallbackInfo const& info);
  Napi::Value num_children(Napi::CallbackInfo const& info);

  Napi::Value get_child(Napi::CallbackInfo const& info);
  // Napi::Value set_child(Napi::CallbackInfo const& info);

  Napi::Value get_value(Napi::CallbackInfo const& info);
  // Napi::Value set_value(Napi::CallbackInfo const& info);

  Napi::Value set_null_mask(Napi::CallbackInfo const& info);
  Napi::Value set_null_count(Napi::CallbackInfo const& info);

  // Napi::Value hasNulls(Napi::CallbackInfo const& info);
  // Napi::Value setNullCount(Napi::CallbackInfo const& info);
  // Napi::Value nullCount(Napi::CallbackInfo const& info);
  // Napi::Value child(Napi::CallbackInfo const& info);
  // Napi::Value numChildren(Napi::CallbackInfo const& info);
};

}  // namespace nv
