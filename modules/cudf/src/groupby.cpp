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
  Napi::Function ctor = DefineClass(env,
                                    "GroupBy",
                                    {
                                      InstanceMethod<&GroupBy::get_groups>("_getGroups"),
                                      // aggregations
                                      InstanceMethod<&GroupBy::argmax>("_argmax"),
                                      InstanceMethod<&GroupBy::argmin>("_argmin"),
                                      InstanceMethod<&GroupBy::count>("_count"),
                                      InstanceMethod<&GroupBy::max>("_max"),
                                      InstanceMethod<&GroupBy::mean>("_mean"),
                                      InstanceMethod<&GroupBy::median>("_median"),
                                      InstanceMethod<&GroupBy::min>("_min"),
                                      InstanceMethod<&GroupBy::nth>("_nth"),
                                      InstanceMethod<&GroupBy::nunique>("_nunique"),
                                      InstanceMethod<&GroupBy::std>("_std"),
                                      InstanceMethod<&GroupBy::sum>("_sum"),
                                      InstanceMethod<&GroupBy::var>("_var"),
                                      InstanceMethod<&GroupBy::quantile>("_quantile"),
                                    });

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
  CallbackArgs args{info};
  rmm::mr::device_memory_resource* mr = args[1];

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

Napi::Value GroupBy::argmax(Napi::CallbackInfo const& info) {
  auto args   = _get_basic_args(info);
  auto values = args.first;
  auto mr     = args.second;
  auto agg    = cudf::make_argmax_aggregation();
  return _single_aggregation(std::move(agg), values, mr, info);
}

Napi::Value GroupBy::argmin(Napi::CallbackInfo const& info) {
  auto args   = _get_basic_args(info);
  auto values = args.first;
  auto mr     = args.second;
  auto agg    = cudf::make_argmin_aggregation();
  return _single_aggregation(std::move(agg), values, mr, info);
}

Napi::Value GroupBy::count(Napi::CallbackInfo const& info) {
  auto args   = _get_basic_args(info);
  auto values = args.first;
  auto mr     = args.second;
  auto agg    = cudf::make_count_aggregation();
  return _single_aggregation(std::move(agg), values, mr, info);
}

Napi::Value GroupBy::max(Napi::CallbackInfo const& info) {
  auto args   = _get_basic_args(info);
  auto values = args.first;
  auto mr     = args.second;
  auto agg    = cudf::make_max_aggregation();
  return _single_aggregation(std::move(agg), values, mr, info);
}

Napi::Value GroupBy::mean(Napi::CallbackInfo const& info) {
  auto args   = _get_basic_args(info);
  auto values = args.first;
  auto mr     = args.second;
  auto agg    = cudf::make_mean_aggregation();
  return _single_aggregation(std::move(agg), values, mr, info);
}

Napi::Value GroupBy::median(Napi::CallbackInfo const& info) {
  auto args   = _get_basic_args(info);
  auto values = args.first;
  auto mr     = args.second;
  auto agg    = cudf::make_median_aggregation();
  return _single_aggregation(std::move(agg), values, mr, info);
}

Napi::Value GroupBy::min(Napi::CallbackInfo const& info) {
  auto args   = _get_basic_args(info);
  auto values = args.first;
  auto mr     = args.second;
  auto agg    = cudf::make_min_aggregation();
  return _single_aggregation(std::move(agg), values, mr, info);
}

Napi::Value GroupBy::nth(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};

  cudf::size_type n = args[0];

  auto values = args[1];
  NODE_CUDA_EXPECT(Table::is_instance(values),
                   "aggregation expects options to have a 'values' table");
  nv::Table* values_table = Table::Unwrap(values.ToObject());

  auto mr = MemoryResource::is_instance(info[2]) ? *MemoryResource::Unwrap(info[2].ToObject())
                                                 : rmm::mr::get_current_device_resource();

  auto agg = cudf::make_nth_element_aggregation(n);

  return _single_aggregation(std::move(agg), values_table, mr, info);
}

Napi::Value GroupBy::nunique(Napi::CallbackInfo const& info) {
  auto args   = _get_basic_args(info);
  auto values = args.first;
  auto mr     = args.second;
  auto agg    = cudf::make_nunique_aggregation();
  return _single_aggregation(std::move(agg), values, mr, info);
}

Napi::Value GroupBy::std(Napi::CallbackInfo const& info) {
  auto args   = _get_basic_args(info);
  auto values = args.first;
  auto mr     = args.second;
  auto agg    = cudf::make_std_aggregation();
  return _single_aggregation(std::move(agg), values, mr, info);
}

Napi::Value GroupBy::sum(Napi::CallbackInfo const& info) {
  auto args   = _get_basic_args(info);
  auto values = args.first;
  auto mr     = args.second;
  auto agg    = cudf::make_sum_aggregation();
  return _single_aggregation(std::move(agg), values, mr, info);
}

Napi::Value GroupBy::var(Napi::CallbackInfo const& info) {
  auto args   = _get_basic_args(info);
  auto values = args.first;
  auto mr     = args.second;
  auto agg    = cudf::make_variance_aggregation();
  return _single_aggregation(std::move(agg), values, mr, info);
}

Napi::Value GroupBy::quantile(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};

  double q = args[0];
  std::vector<double> qs{q};

  auto values = args[1];
  NODE_CUDA_EXPECT(Table::is_instance(values),
                   "GroupBy quantile_agg expects options to have a 'values' table");
  nv::Table* values_table = Table::Unwrap(values.ToObject());

  // TODO handle both optional
  cudf::interpolation interpolation = args[2];

  auto mr = MemoryResource::is_instance(info[3]) ? *MemoryResource::Unwrap(info[3].ToObject())
                                                 : rmm::mr::get_current_device_resource();

  auto agg = cudf::make_quantile_aggregation(qs, interpolation);

  return _single_aggregation(std::move(agg), values_table, mr, info);
}

std::pair<nv::Table*, rmm::mr::device_memory_resource*> GroupBy::_get_basic_args(
  Napi::CallbackInfo const& info) {
  CallbackArgs args{info};

  auto values = args[0];
  NODE_CUDA_EXPECT(Table::is_instance(values), "aggregation expects to have a 'values' table");

  rmm::mr::device_memory_resource* mr = args[1];

  return std::pair<Table*, rmm::mr::device_memory_resource*>(Table::Unwrap(values.ToObject()), mr);
}

Napi::Value GroupBy::_single_aggregation(std::unique_ptr<cudf::aggregation> agg,
                                         const nv::Table* const values_table,
                                         rmm::mr::device_memory_resource* const mr,
                                         Napi::CallbackInfo const& info) {
  std::vector<cudf::groupby::aggregation_request> requests;

  for (cudf::size_type i = 0; i < values_table->num_columns(); ++i) {
    auto request   = cudf::groupby::aggregation_request();
    request.values = values_table->get_column(i).view();
    request.aggregations.push_back(std::move(agg));
    requests.emplace_back(std::move(request));
  }

  auto result = groupby_->aggregate(requests, mr);

  auto result_keys = Table::New(std::move(result.first));

  auto result_cols = Napi::Array::New(info.Env(), result.second.size());
  for (size_t i = 0; i < result.second.size(); ++i) {
    result_cols.Set(i, Column::New(std::move(result.second[i].results[0]))->Value());
  }

  auto obj = Napi::Object::New(info.Env());
  obj.Set("keys", result_keys);
  obj.Set("cols", result_cols);

  return obj;
}

}  // namespace nv
