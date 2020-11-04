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
#include "cudf/utilities/traits.hpp"
#include "node_cudf/utilities/cpp_to_napi.hpp"
#include "node_cudf/utilities/error.hpp"
#include "node_cudf/utilities/napi_to_cpp.hpp"
#include "nv_node/utilities/napi_to_cpp.hpp"

#include <node_cuda/utilities/cpp_to_napi.hpp>
#include <node_cuda/utilities/napi_to_cpp.hpp>

#include <node_rmm/device_buffer.hpp>
#include <node_rmm/utilities/napi_to_cpp.hpp>

#include <nv_node/macros.hpp>
#include <nv_node/utilities/args.hpp>

#include <cudf/table/table.hpp>
#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/types.hpp>

#include <rmm/device_buffer.hpp>

#include <napi.h>

namespace nv {

//
// Public API
//

Napi::FunctionReference Table::constructor;

Napi::Object Table::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor =
    DefineClass(env,
                "Table",
                {
                  InstanceAccessor("numColumns", &Table::num_columns, nullptr, napi_enumerable),
                  InstanceAccessor("numRows", &Table::num_rows, nullptr, napi_enumerable),
                  // InstanceMethod("select", &Table::select),
                  InstanceMethod("getColumn", &Table::get_column),
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

Table::Table(CallbackArgs const& args) : Napi::ObjectWrap<Table>(args) {
  NODE_CUDA_EXPECT(args.IsConstructCall(), "Table constructor requires 'new'");

  if (args.Length() != 1 || !args[0].IsObject()) { return; }

  Napi::Object props = args[0];

  Napi::Array columns = props.Has("columns")  //
                           ? props.Get("columns").As<Napi::Array>()
                           : Napi::Array::New(Env(), 0);

  Initialize(columns);
}

void Table::Initialize(Napi::Array const& columns) {
  num_columns_ = columns.Length();
  if(num_columns_ > 0u){
    num_rows_ = nv::Column::Unwrap(columns.Get(0u).As<Napi::Object>())->size();
    for (auto i = 1; i < columns.Length(); ++i) {
      NODE_CUDA_EXPECT(
        (nv::Column::Unwrap(columns.Get(i).As<Napi::Object>())->size() == num_rows_),
        "All Columns must be of same length"
      );
    }
  }
  
  columns_.Reset(columns, 1);
}

void Table::Finalize(Napi::Env env) {
  columns_.Reset();
}

cudf::table_view Table::view() const {
  auto columns = columns_.Value().As<Napi::Array>();

  // Create views of children
  std::vector<cudf::column_view> child_views;
  child_views.reserve(columns.Length());
  for (auto i = 0; i < columns.Length(); ++i) {
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
  for (auto i = 0; i < columns.Length(); ++i) {
    auto child = columns.Get(i).As<Napi::Object>();
    child_views.emplace_back(*Column::Unwrap(child));
  }

  return cudf::mutable_table_view{child_views};
}

//
// Private API
//
Napi::Value Table::num_columns(Napi::CallbackInfo const& info) { return CPPToNapi(info)(num_columns()); }

Napi::Value Table::num_rows(Napi::CallbackInfo const& info) { return CPPToNapi(info)(num_rows()); }

Napi::Value Table::get_column(Napi::CallbackInfo const& info) {
  return columns_.Value().Get(CallbackArgs{info}[0].operator cudf::size_type());
}


}  // namespace nv
