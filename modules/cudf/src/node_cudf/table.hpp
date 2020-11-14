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

#include <nv_node/utilities/args.hpp>

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>


#include <napi.h>

namespace nv {

/**
 * @brief An owning wrapper around a cudf::Table.
 *
 */
class Table : public Napi::ObjectWrap<Table> {
 public:
  /**
   * @brief Initialize and export the Table JavaScript constructor and prototype.
   *
   * @param env The active JavaScript environment.
   * @param exports The exports object to decorate.
   * @return Napi::Object The decorated exports object.
   */
  static Napi::Object Init(Napi::Env env, Napi::Object exports);

  /**
   * @brief Construct a new Table instance from existing device memory.
   *
   * @param children Array of child columns
   */
  static Napi::Object New(Napi::Array const& columns = {});

  /**
   * @brief Construct a new Column instance from JavaScript.
   *
   */
  Table(CallbackArgs const& args);

  /**
   * @brief Initialize the Table instance created by either C++ or JavaScript.
   *
   * @param data The Table's data.
   * @param size The number of elements in the column.
   * @param type The element data type.
   */
  void Initialize(Napi::Object const& data, cudf::size_type size, Napi::Object const& type) {
    Initialize(Napi::Array::New(Env(), 0));
  }

  /**
   * @brief Initialize the Table instance created by either C++ or JavaScript.
   *
   * @param children Array of columns
   */
  void Initialize(Napi::Array const& columns = {});

  /**
   * @brief Destructor called when the JavaScript VM garbage collects this Column
   * instance.
   *
   * @param env The active JavaScript environment.
   */
  void Finalize(Napi::Env env) override;

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
   * @brief Implicit conversion operator to a `mutable_table_view `.
   *
   * This allows pasing a `table` object into a function that accepts a
   *`mutable_table_view `. The conversion is automatic.
   * @return cudf::mutable_table_view  Mutable, non-owning `mutable_table_view `
   */
  operator cudf::mutable_table_view () { return this->mutable_view(); };

 private:
  static Napi::FunctionReference constructor;

  cudf::size_type num_columns_{};      ///< The number of columns in the table
  cudf::size_type num_rows_{};         ///< The number of rows
  Napi::Reference<Napi::Array> columns_{}; ///< columns of table

  Napi::Value num_columns(Napi::CallbackInfo const& info);
  Napi::Value num_rows(Napi::CallbackInfo const& info);
  Napi::Value select(Napi::CallbackInfo const& info);
  Napi::Value get_column(Napi::CallbackInfo const& info);

};

}  // namespace nv
