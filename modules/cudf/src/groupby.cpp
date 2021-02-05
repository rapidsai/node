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

#include "node_cudf/groupby.hpp"
#include "node_cudf/table.hpp"
#include "node_cudf/utilities/error.hpp"
#include "node_cudf/utilities/napi_to_cpp.hpp"

#include <cudf/groupby.hpp>
#include <cudf/types.hpp>
#include <node_cuda/utilities/error.hpp>

#include <napi.h>

namespace nv {

//
// Public API
//

Napi::FunctionReference GroupBy::constructor;

Napi::Object GroupBy::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor =
    DefineClass(env, "GroupBy", {InstanceMethod<&GroupBy::get_groups>("_getGroups")});

  GroupBy::constructor = Napi::Persistent(ctor);
  GroupBy::constructor.SuppressDestruct();
  exports.Set("GroupBy", ctor);

  return exports;
}

Napi::Object GroupBy::New() {
  auto inst = GroupBy::constructor.New({});
  return inst;
}

GroupBy::GroupBy(CallbackArgs const& args) : Napi::ObjectWrap<GroupBy>(args) {
  using namespace cudf;

  Napi::Object props = args[0];
  NODE_CUDA_EXPECT(props.Has("keys"), "GroupBy constructor expects options to have a 'keys' field");

  if (!Table::is_instance(props.Get("keys"))) {
    NAPI_THROW(Napi::Error::New(args.Env(), "GroupBy constructor 'keys' field expects a Table."));
  }

  cudf::table_view table_view = Table::Unwrap(props.Get("keys").ToObject())->view();

  auto null_handling = null_policy::EXCLUDE;
  if (props.Has("include_nulls")) { null_handling = NapiToCPP(props.Get("include_nulls")); }

  auto keys_are_sorted = sorted::NO;
  if (props.Has("keys_are_sorted")) { keys_are_sorted = NapiToCPP(props.Get("keys_are_sorted")); }

  std::vector<order> column_order =
    NapiToCPP{props.Has("column_order") ? props["column_order"] : args.Env().Null()};

  std::vector<null_order> null_precedence =
    NapiToCPP{props.Has("null_precedence") ? props["null_precedence"] : args.Env().Null()};

  groupby_.reset(new groupby::groupby(
    table_view, null_handling, keys_are_sorted, column_order, null_precedence));
}

void GroupBy::Finalize(Napi::Env env) { this->groupby_.reset(nullptr); }

//
// Private API
//

Napi::Value GroupBy::get_groups(Napi::CallbackInfo const& info) {
  auto values = info[0];
  auto mr     = MemoryResource::is_instance(info[1]) ? *MemoryResource::Unwrap(info[1].ToObject())
                                                     : rmm::mr::get_current_device_resource();

  cudf::table_view table{cudf::table{}};
  if (Table::is_instance(values)) { table = *Table::Unwrap(values.ToObject()); }
  auto groups = groupby_->get_groups(table, mr);

  auto result = Napi::Object::New(info.Env());
  result.Set("keys", Table::New(std::move(groups.keys)));

  auto const& offsets = groups.offsets;
  result.Set("offsets", CPPToNapi(info)(std::make_tuple(offsets.data(), offsets.size())));

  if (groups.values != nullptr) { result.Set("values", Table::New(std::move(groups.values))); }
  return result;
}

}  // namespace nv
