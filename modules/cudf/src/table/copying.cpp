// Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <node_cudf/table.hpp>
#include <node_cudf/utilities/napi_to_cpp.hpp>

#include <cudf/copying.hpp>
#include <cudf/table/table_view.hpp>

#include <memory>

namespace nv {

ObjectUnwrap<Table> Table::gather(Column const& gather_map,
                                  cudf::out_of_bounds_policy bounds_policy,
                                  rmm::mr::device_memory_resource* mr) const {
  return Table::New(cudf::gather(cudf::table_view{{*this}}, gather_map, bounds_policy, mr));
}

ObjectUnwrap<Table> Table::scatter(
  std::vector<std::reference_wrapper<const cudf::scalar>> const& source,
  Column const& indices,
  bool check_bounds,
  rmm::mr::device_memory_resource* mr) const {
  try {
    return Table::New(cudf::scatter(source, indices.view(), this->view(), check_bounds, mr));
  } catch (cudf::logic_error const& err) { NAPI_THROW(Napi::Error::New(Env(), err.what())); }
}

ObjectUnwrap<Table> Table::scatter(Table const& source,
                                   Column const& indices,
                                   bool check_bounds,
                                   rmm::mr::device_memory_resource* mr) const {
  try {
    return Table::New(cudf::scatter(source.view(), indices.view(), this->view(), check_bounds, mr));
  } catch (cudf::logic_error const& err) { NAPI_THROW(Napi::Error::New(Env(), err.what())); }
}

Napi::Value Table::gather(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  if (!Column::is_instance(args[0])) {
    throw Napi::Error::New(info.Env(), "gather selection argument expects a Column");
  }
  auto& selection = *Column::Unwrap(args[0]);
  if (selection.type().id() == cudf::type_id::BOOL8) {
    return this->apply_boolean_mask(selection)->Value();
  }
  return this->gather(selection)->Value();
}

Napi::Value Table::scatter_scalar(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};

  if (args.Length() != 3 and args.Length() != 4) {
    NAPI_THROW(Napi::Error::New(info.Env(),
                                "scatter_scalar expects a vector of scalars, a Column, and "
                                "optionally a bool and memory resource"));
  }

  if (!args[0].IsArray()) {
    throw Napi::Error::New(info.Env(), "scatter_scalar source argument expects an array");
  }
  auto const source_array = args[0].As<Napi::Array>();
  std::vector<std::reference_wrapper<const cudf::scalar>> source{};

  for (uint32_t i = 0; i < source_array.Length(); ++i) {
    if (!Scalar::is_instance(source_array.Get(i))) {
      throw Napi::Error::New(info.Env(),
                             "scatter_scalar source argument expects an array of scalars");
    }
    source.push_back(std::ref<const cudf::scalar>(*Scalar::Unwrap(source_array.Get(i).ToObject())));
  }

  if (!Column::is_instance(args[1])) {
    throw Napi::Error::New(info.Env(), "scatter_scalar indices argument expects a Column");
  }
  auto& indices                       = *Column::Unwrap(args[1]);
  bool check_bounds                   = args[2];
  rmm::mr::device_memory_resource* mr = args[3];
  return scatter(source, indices, check_bounds, mr)->Value();
}

Napi::Value Table::scatter_table(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};

  if (args.Length() != 3 and args.Length() != 4) {
    NAPI_THROW(Napi::Error::New(
      info.Env(),
      "scatter_table expects a Table, a Column, and optionally a bool and memory resource"));
  }

  if (!Table::is_instance(args[0])) {
    throw Napi::Error::New(info.Env(), "scatter_table source argument expects a Table");
  }
  auto& source = *Table::Unwrap(args[0]);

  if (!Column::is_instance(args[1])) {
    throw Napi::Error::New(info.Env(), "scatter_table indices argument expects a Column");
  }
  auto& indices = *Column::Unwrap(args[1]);

  bool check_bounds                   = args[2];
  rmm::mr::device_memory_resource* mr = args[3];
  return scatter(source, indices, check_bounds, mr)->Value();
}

}  // namespace nv
