// Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include "node_cudf/table.hpp"
#include "node_cudf/column.hpp"
#include "node_cudf/utilities/error.hpp"
#include "node_cudf/utilities/napi_to_cpp.hpp"

#include <cudf/column/column.hpp>
#include <cudf/sorting.hpp>
#include <cudf/types.hpp>

#include <napi.h>

namespace nv {

//
// Public API
//

Napi::Function Table::Init(Napi::Env const& env, Napi::Object exports) {
  return DefineClass(env,
                     "Table",
                     {
                       InstanceAccessor<&Table::num_columns>("numColumns"),
                       InstanceAccessor<&Table::num_rows>("numRows"),
                       InstanceMethod<&Table::dispose>("dispose"),
                       InstanceMethod<&Table::scatter_scalar>("scatterScalar"),
                       InstanceMethod<&Table::scatter_table>("scatterTable"),
                       InstanceMethod<&Table::gather>("gather"),
                       InstanceMethod<&Table::apply_boolean_mask>("applyBooleanMask"),
                       InstanceMethod<&Table::get_column>("getColumnByIndex"),
                       InstanceMethod<&Table::to_arrow>("toArrow"),
                       InstanceMethod<&Table::order_by>("orderBy"),
                       StaticMethod<&Table::read_csv>("readCSV"),
                       InstanceMethod<&Table::write_csv>("writeCSV"),
                       StaticMethod<&Table::read_parquet>("readParquet"),
                       InstanceMethod<&Table::write_parquet>("writeParquet"),
                       StaticMethod<&Table::read_orc>("readORC"),
                       InstanceMethod<&Table::write_orc>("writeORC"),
                       StaticMethod<&Table::from_arrow>("fromArrow"),
                       InstanceMethod<&Table::drop_nans>("dropNans"),
                       InstanceMethod<&Table::drop_nulls>("dropNulls"),
                       InstanceMethod<&Table::drop_duplicates>("dropDuplicates"),
                       InstanceMethod<&Table::interleave_columns>("interleaveColumns"),
                       StaticMethod<&Table::concat>("concat"),
                       StaticMethod<&Table::full_join>("fullJoin"),
                       StaticMethod<&Table::inner_join>("innerJoin"),
                       StaticMethod<&Table::left_join>("leftJoin"),
                       StaticMethod<&Table::left_semi_join>("leftSemiJoin"),
                       StaticMethod<&Table::left_anti_join>("leftAntiJoin"),
                       InstanceMethod<&Table::explode>("explode"),
                       InstanceMethod<&Table::explode_position>("explodePosition"),
                       InstanceMethod<&Table::explode_outer>("explodeOuter"),
                       InstanceMethod<&Table::explode_outer_position>("explodeOuterPosition"),
                     });
}

Table::wrapper_t Table::New(Napi::Env const& env, Napi::Array const& columns) {
  auto opts = Napi::Object::New(env);
  opts.Set("columns", columns);
  return EnvLocalObjectWrap<Table>::New(env, {opts});
}

Table::wrapper_t Table::New(Napi::Env const& env, std::unique_ptr<cudf::table> table) {
  auto contents = table->release();
  auto columns  = Napi::Array::New(env, contents.size());
  for (auto i = 0u; i < columns.Length(); ++i) {
    columns.Set(i, Column::New(env, std::move(contents[i])));
  }
  return New(env, columns);
}

Table::Table(CallbackArgs const& args) : EnvLocalObjectWrap<Table>(args) {
  // std::cout << "Table::Table" << std::endl;
  auto env = args.Env();
  NODE_CUDF_EXPECT(args.IsConstructCall(), "Table constructor requires 'new'", env);

  if (args.Length() != 1 || !args[0].IsObject()) { return; }

  NapiToCPP::Object props = args[0];

  num_rows_    = 0;
  num_columns_ = 0;

  if (props.Get("columns").IsArray()) {
    auto cols    = props.Get("columns").As<Napi::Array>();
    num_columns_ = cols.Length();
    for (auto i = 0; i < num_columns_; ++i) {
      columns_.push_back(Napi::Persistent(Column::wrapper_t{cols.Get(i).ToObject()}));
    }
  }

  if (num_columns_ > 0) {
    num_rows_ = get_column(0).size();
    for (auto i = 1; i < num_columns_; ++i) {
      NODE_CUDF_EXPECT(
        get_column(i).size() == num_rows_, "All Columns must be of same length", env);
    }
  }
}

void Table::Finalize(Napi::Env env) {
  //
}

void Table::dispose(Napi::Env env) {
  if (columns_.size() > 0) {
    for (auto& child : columns_) { child.Value()->dispose(env); }
    columns_.clear();
  }
  num_rows_    = 0;
  num_columns_ = 0;
  disposed_    = true;
}

void Table::dispose(Napi::CallbackInfo const& info) { dispose(info.Env()); }

cudf::table_view Table::view() const {
  // Create views of children
  std::vector<cudf::column_view> child_views;
  child_views.reserve(num_columns_);
  for (auto i = 0; i < num_columns_; ++i) { child_views.push_back(columns_[i].Value()->view()); }

  return cudf::table_view{child_views};
}

cudf::mutable_table_view Table::mutable_view() {
  // Create views of children
  std::vector<cudf::mutable_column_view> child_views;
  child_views.reserve(num_columns_);
  for (auto i = 0; i < num_columns_; ++i) {
    child_views.push_back(columns_[i].Value()->mutable_view());
  }

  return cudf::mutable_table_view{child_views};
}

//
// Private API
//
Napi::Value Table::num_columns(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(num_columns());
}

Napi::Value Table::num_rows(Napi::CallbackInfo const& info) { return CPPToNapi(info)(num_rows()); }

Napi::Value Table::get_column(Napi::CallbackInfo const& info) {
  cudf::size_type i = CallbackArgs{info}[0];
  if (i >= num_columns_) { throw Napi::Error::New(info.Env(), "Column index out of bounds"); }
  return columns_[i].Value();
}

Napi::Value Table::order_by(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};

  NODE_CUDF_EXPECT(args[0].IsArray(), "order_by ascending argument expects an array", args.Env());
  NODE_CUDF_EXPECT(args[1].IsArray(), "order_by null_order argument expects an array", args.Env());

  std::vector<bool> ascending  = args[0];
  std::vector<bool> null_order = args[1];

  NODE_CUDF_EXPECT(ascending.size() == null_order.size(),
                   "ascending and null_order must be the same size",
                   args.Env());

  auto table_view = view();

  std::vector<cudf::order> column_order;
  column_order.reserve(ascending.size());
  for (auto i : ascending) {
    if (i) {
      column_order.push_back(cudf::order::ASCENDING);
    } else {
      column_order.push_back(cudf::order::DESCENDING);
    }
  }

  std::vector<cudf::null_order> null_precedece;
  null_precedece.reserve(null_order.size());
  for (auto i : null_order) {
    if (i) {
      null_precedece.push_back(cudf::null_order::BEFORE);
    } else {
      null_precedece.push_back(cudf::null_order::AFTER);
    }
  }

  std::unique_ptr<cudf::column> result =
    cudf::sorted_order(table_view, column_order, null_precedece);

  return Column::New(info.Env(), std::move(result))->Value();
}

}  // namespace nv
