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

Napi::FunctionReference Table::constructor;

Napi::Object Table::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor = DefineClass(env,
                                    "Table",
                                    {
                                      InstanceAccessor<&Table::num_columns>("numColumns"),
                                      InstanceAccessor<&Table::num_rows>("numRows"),
                                      InstanceMethod<&Table::scatter_scalar>("scatterScalar"),
                                      InstanceMethod<&Table::scatter_table>("scatterTable"),
                                      InstanceMethod<&Table::gather>("gather"),
                                      InstanceMethod<&Table::get_column>("getColumnByIndex"),
                                      InstanceMethod<&Table::to_arrow>("toArrow"),
                                      InstanceMethod<&Table::order_by>("orderBy"),
                                      StaticMethod<&Table::read_csv>("readCSV"),
                                      InstanceMethod<&Table::write_csv>("writeCSV"),
                                      InstanceMethod<&Table::drop_nans>("drop_nans"),
                                      InstanceMethod<&Table::drop_nulls>("drop_nulls"),
                                    });

  Table::constructor = Napi::Persistent(ctor);
  Table::constructor.SuppressDestruct();
  exports.Set("Table", ctor);

  return exports;
}

Napi::Object Table::New(Napi::Array const& columns) {
  auto inst = Table::constructor.New({});
  Table::Unwrap(inst)->Initialize(columns);
  return inst;
}

Napi::Object Table::New(std::unique_ptr<cudf::table> table) {
  auto inst     = Table::constructor.New({});
  auto contents = table->release();
  auto columns  = Napi::Array::New(Table::constructor.Env(), contents.size());
  for (auto i = 0u; i < columns.Length(); ++i) {
    columns.Set(i, Column::New(std::move(contents[i]))->Value());
  }
  Table::Unwrap(inst)->Initialize(columns);
  return inst;
}

Table::Table(CallbackArgs const& args) : Napi::ObjectWrap<Table>(args) {
  NODE_CUDF_EXPECT(args.IsConstructCall(), "Table constructor requires 'new'");

  if (args.Length() != 1 || !args[0].IsObject()) { return; }

  Napi::Object props = args[0];

  Napi::Array columns = props.Has("columns")  //
                          ? props.Get("columns").As<Napi::Array>()
                          : Napi::Array::New(Env(), 0);

  Initialize(columns);
}

void Table::Initialize(Napi::Array const& columns) {
  num_columns_ = columns.Length();
  if (num_columns_ > 0) {
    num_rows_ = nv::Column::Unwrap(columns.Get(0u).As<Napi::Object>())->size();
    for (auto i = 1u; i < columns.Length(); ++i) {
      NODE_CUDF_EXPECT((nv::Column::Unwrap(columns.Get(i).As<Napi::Object>())->size() == num_rows_),
                       "All Columns must be of same length");
    }
  }

  columns_.Reset(columns, 1);
}

void Table::Finalize(Napi::Env env) { columns_.Reset(); }

cudf::table_view Table::view() const {
  auto columns = columns_.Value().As<Napi::Array>();

  // Create views of children
  std::vector<cudf::column_view> child_views;
  child_views.reserve(columns.Length());
  for (auto i = 0u; i < columns.Length(); ++i) {
    auto child = columns.Get(i).As<Napi::Object>();
    child_views.emplace_back(*Column::Unwrap(child));
  }

  return cudf::table_view{child_views};
}

cudf::mutable_table_view Table::mutable_view() {
  auto columns = columns_.Value().As<Napi::Array>();

  // Create views of children
  std::vector<cudf::mutable_column_view> child_views;
  child_views.reserve(columns.Length());
  for (auto i = 0u; i < columns.Length(); ++i) {
    auto child = columns.Get(i).As<Napi::Object>();
    child_views.emplace_back(*Column::Unwrap(child));
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
  return columns_.Value().Get(i);
}

Napi::Value Table::order_by(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};

  NODE_CUDF_EXPECT(args[0].IsArray(), "order_by ascending argument expects an array");
  NODE_CUDF_EXPECT(args[1].IsArray(), "order_by null_order argument expects an array");

  std::vector<bool> ascending  = args[0];
  std::vector<bool> null_order = args[1];

  NODE_CUDF_EXPECT(ascending.size() == null_order.size(),
                   "ascending and null_order must be the same size");

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

  return Column::New(std::move(result))->Value();
}

}  // namespace nv
