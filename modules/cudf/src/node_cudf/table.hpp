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

#include <node_cudf/column.hpp>

#include <node_rmm/device_buffer.hpp>

#include <nv_node/objectwrap.hpp>
#include <nv_node/utilities/args.hpp>

#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <napi.h>

namespace nv {

/**
 * @brief An owning wrapper around a cudf::Table.
 *
 */
struct Table : public EnvLocalObjectWrap<Table> {
  /**
   * @brief Initialize and export the Table JavaScript constructor and prototype.
   *
   * @param env The active JavaScript environment.
   * @param exports The exports object to decorate.
   * @return Napi::Function The Table constructor function.
   */
  static Napi::Function Init(Napi::Env const& env, Napi::Object exports);

  /**
   * @brief Construct a new Table instance from existing device memory.
   *
   * @param children Array of child columns
   * @return wrapper_t The new Table instance
   */
  static wrapper_t New(Napi::Env const& env, Napi::Array const& columns = {});

  /**
   * @brief Construct a new Table instance from an existing libcudf Table
   *
   * @param table The libcudf Table to adapt
   * @return wrapper_t The new Table instance
   */
  static wrapper_t New(Napi::Env const& env, std::unique_ptr<cudf::table> table);

  /**
   * @brief Construct a new Column instance from JavaScript.
   *
   */
  Table(CallbackArgs const& args);

  /**
   * @brief Returns the number of columns in the table
   */
  cudf::size_type num_columns() const noexcept { return num_columns_; }

  /**
   * @brief Returns the number of columns in the table
   */
  cudf::size_type num_rows() const noexcept { return num_rows_; }

  /**
   * @brief Creates an immutable, non-owning view of the table
   *
   * @return cudf::table_view The immutable, non-owning view
   */
  cudf::table_view view() const;

  /**
   * @brief Creates a mutable, non-owning view of the table
   *
   * @return cudf::mutable_table_view The mutable, non-owning view
   */
  cudf::mutable_table_view mutable_view();

  /**
   * @brief Implicit conversion operator to a `table_view`.
   *
   * This allows passing a `table` object directly into a function that
   * requires a `table_view`. The conversion is automatic.
   *
   * @return cudf::table_view Immutable, non-owning `table_view`
   */
  operator cudf::table_view() const { return this->view(); };

  /**
   * @brief Implicit conversion operator to a `mutable_table_view`.
   *
   * This allows pasing a `table` object into a function that accepts a
   *`mutable_table_view `. The conversion is automatic.
   * @return cudf::mutable_table_view  Mutable, non-owning `mutable_table_view`
   */
  operator cudf::mutable_table_view() { return this->mutable_view(); };

  /**
   * @brief Returns a const reference to the specified column
   *
   * @throws std::out_of_range
   * If i is out of the range [0, num_columns)
   *
   * @param i Index of the desired column
   * @return A const reference to the desired column
   */
  Column const& get_column(cudf::size_type i) const {
    return *Column::Unwrap(columns_.Value().Get(i).ToObject());
  }

  // table/reshape.cpp
  Column::wrapper_t interleave_columns(
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  // table/stream_compaction.cpp
  Table::wrapper_t apply_boolean_mask(
    Column const& boolean_mask,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  Table::wrapper_t drop_nulls(
    std::vector<cudf::size_type> keys,
    cudf::size_type threshold,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  Table::wrapper_t drop_nans(
    std::vector<cudf::size_type> keys,
    cudf::size_type threshold,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  Table::wrapper_t drop_duplicates(
    std::vector<cudf::size_type> keys,
    cudf::duplicate_keep_option keep,
    bool nulls_equal,
    bool is_nulls_first,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  // table/copying.cpp
  Table::wrapper_t gather(
    Column const& gather_map,
    cudf::out_of_bounds_policy bounds_policy = cudf::out_of_bounds_policy::DONT_CHECK,
    rmm::mr::device_memory_resource* mr      = rmm::mr::get_current_device_resource()) const;

  Table::wrapper_t scatter(
    std::vector<std::reference_wrapper<const cudf::scalar>> const& source,
    Column const& indices,
    bool check_bounds                   = false,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  Table::wrapper_t scatter(
    Table const& source,
    Column const& indices,
    bool check_bounds                   = false,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  // table/join.cpp
  static std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
                   std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
  full_join(Napi::Env const& env,
            Table const& left,
            Table const& right,
            bool null_equality,
            rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  static std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
                   std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
  inner_join(Napi::Env const& env,
             Table const& left,
             Table const& right,
             bool null_equality,
             rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  static std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
                   std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
  left_join(Napi::Env const& env,
            Table const& left,
            Table const& right,
            bool null_equality,
            rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  static std::unique_ptr<rmm::device_uvector<cudf::size_type>> left_semi_join(
    Napi::Env const& env,
    Table const& left,
    Table const& right,
    bool null_equality,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  static std::unique_ptr<rmm::device_uvector<cudf::size_type>> left_anti_join(
    Napi::Env const& env,
    Table const& left,
    Table const& right,
    bool null_equality,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

 private:
  cudf::size_type num_columns_{};           ///< The number of columns in the table
  cudf::size_type num_rows_{};              ///< The number of rows
  Napi::Reference<Napi::Array> columns_{};  ///< columns of table

  Napi::Value num_columns(Napi::CallbackInfo const& info);
  Napi::Value num_rows(Napi::CallbackInfo const& info);
  Napi::Value select(Napi::CallbackInfo const& info);
  Napi::Value get_column(Napi::CallbackInfo const& info);
  // table/reshape.cpp
  Napi::Value interleave_columns(Napi::CallbackInfo const& info);
  // table/stream_compaction.cpp
  Napi::Value drop_nulls(Napi::CallbackInfo const& info);
  Napi::Value drop_nans(Napi::CallbackInfo const& info);
  Napi::Value drop_duplicates(Napi::CallbackInfo const& info);
  // table/copying.cpp
  Napi::Value gather(Napi::CallbackInfo const& info);
  Napi::Value scatter_scalar(Napi::CallbackInfo const& info);
  Napi::Value scatter_table(Napi::CallbackInfo const& info);
  // table/concatenate.cpp
  static Napi::Value concat(Napi::CallbackInfo const& info);
  // table/join.cpp
  static Napi::Value full_join(Napi::CallbackInfo const& info);
  static Napi::Value inner_join(Napi::CallbackInfo const& info);
  static Napi::Value left_join(Napi::CallbackInfo const& info);
  static Napi::Value left_semi_join(Napi::CallbackInfo const& info);
  static Napi::Value left_anti_join(Napi::CallbackInfo const& info);

  static Napi::Value read_csv(Napi::CallbackInfo const& info);
  void write_csv(Napi::CallbackInfo const& info);

  static Napi::Value read_parquet(Napi::CallbackInfo const& info);
  static Napi::Value from_arrow(Napi::CallbackInfo const& info);

  Napi::Value to_arrow(Napi::CallbackInfo const& info);
  Napi::Value order_by(Napi::CallbackInfo const& info);
};

}  // namespace nv
