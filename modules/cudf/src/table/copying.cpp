// Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <functional>
#include <memory>

namespace nv {

Table::wrapper_t Table::gather(Column const& gather_map,
                               cudf::out_of_bounds_policy bounds_policy,
                               rmm::mr::device_memory_resource* mr) const {
  return Table::New(Env(), cudf::gather(cudf::table_view{{*this}}, gather_map, bounds_policy, mr));
}

Table::wrapper_t Table::scatter(
  std::vector<std::reference_wrapper<const cudf::scalar>> const& source,
  Column const& indices,
  bool check_bounds,
  rmm::mr::device_memory_resource* mr) const {
  try {
    return Table::New(Env(), cudf::scatter(source, indices, *this, check_bounds, mr));
  } catch (std::exception const& e) { throw Napi::Error::New(Env(), e.what()); }
}

Table::wrapper_t Table::scatter(Table const& source,
                                Column const& indices,
                                bool check_bounds,
                                rmm::mr::device_memory_resource* mr) const {
  try {
    return Table::New(Env(), cudf::scatter(source, indices, *this, check_bounds, mr));
  } catch (std::exception const& e) { throw Napi::Error::New(Env(), e.what()); }
}

Napi::Value Table::gather(Napi::CallbackInfo const& info) {
  using namespace cudf;

  CallbackArgs args{info};
  if (!Column::IsInstance(args[0])) {
    throw Napi::Error::New(info.Env(), "gather selection argument expects a Column");
  }

  Column::wrapper_t selection = args[0].ToObject();

  auto oob_policy =
    args[1].ToBoolean() ? out_of_bounds_policy::NULLIFY : out_of_bounds_policy::DONT_CHECK;

  rmm::mr::device_memory_resource* mr = args[2];

  return this->gather(selection, oob_policy, mr);
}

Napi::Value Table::apply_boolean_mask(Napi::CallbackInfo const& info) {
  using namespace cudf;

  CallbackArgs args{info};
  if (!Column::IsInstance(args[0])) {
    throw Napi::Error::New(info.Env(), "apply_boolean_mask selection argument expects a Column");
  }

  Column::wrapper_t selection = args[0].ToObject();

  rmm::mr::device_memory_resource* mr = args[1];

  return this->apply_boolean_mask(selection, mr);
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
    if (!Scalar::IsInstance(source_array.Get(i))) {
      throw Napi::Error::New(info.Env(),
                             "scatter_scalar source argument expects an array of scalars");
    }
    source.push_back(std::ref<const cudf::scalar>(*Scalar::Unwrap(source_array.Get(i).ToObject())));
  }

  if (!Column::IsInstance(args[1])) {
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

  if (!Table::IsInstance(args[0])) {
    throw Napi::Error::New(info.Env(), "scatter_table source argument expects a Table");
  }
  auto& source = *Table::Unwrap(args[0]);

  if (!Column::IsInstance(args[1])) {
    throw Napi::Error::New(info.Env(), "scatter_table indices argument expects a Column");
  }
  auto& indices = *Column::Unwrap(args[1]);

  bool check_bounds                   = args[2];
  rmm::mr::device_memory_resource* mr = args[3];
  return scatter(source, indices, check_bounds, mr)->Value();
}

}  // namespace nv
