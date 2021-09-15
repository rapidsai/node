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

Napi::Function GroupBy::Init(Napi::Env const& env, Napi::Object exports) {
  return DefineClass(env,
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
}

GroupBy::wrapper_t GroupBy::New(Napi::Env const& env) {
  return EnvLocalObjectWrap<GroupBy>::New(env, {});
}

GroupBy::GroupBy(CallbackArgs const& args) : EnvLocalObjectWrap<GroupBy>(args) {
  using namespace cudf;

  NapiToCPP::Object props = args[0];
  NODE_CUDA_EXPECT(props.Has("keys"), "GroupBy constructor expects options to have a 'keys' field");

  if (!Table::IsInstance(props.Get("keys"))) {
    NAPI_THROW(Napi::Error::New(args.Env(), "GroupBy constructor 'keys' field expects a Table."));
  }

  Table::wrapper_t table = props.Get("keys").ToObject();

  auto null_handling = null_policy::EXCLUDE;
  if (props.Has("include_nulls")) { null_handling = props.Get("include_nulls"); }

  auto keys_are_sorted = sorted::NO;
  if (props.Has("keys_are_sorted")) { keys_are_sorted = props.Get("keys_are_sorted"); }

  std::vector<order> column_order =
    props.Has("column_order") ? props.Get("column_order") : NapiToCPP{args.Env().Null()};

  std::vector<null_order> null_precedence =
    props.Has("null_precedence") ? props.Get("null_precedence") : NapiToCPP{args.Env().Null()};

  groupby_.reset(new groupby::groupby(
    table->view(), null_handling, keys_are_sorted, column_order, null_precedence));
}

//
// Private API
//

Napi::Value GroupBy::get_groups(Napi::CallbackInfo const& info) {
  auto values = info[0];
  auto env    = info.Env();
  CallbackArgs args{info};
  rmm::mr::device_memory_resource* mr = args[1];

  cudf::table_view table{cudf::table{}};
  if (Table::IsInstance(values)) { table = *Table::Unwrap(values.ToObject()); }
  auto groups = groupby_->get_groups(table, mr);

  auto result = Napi::Object::New(env);
  result.Set("keys", Table::New(env, std::move(groups.keys)));

  auto const& offsets = groups.offsets;
  result.Set("offsets", CPPToNapi(info)(std::make_tuple(offsets.data(), offsets.size())));

  if (groups.values != nullptr) {  //
    result.Set("values", Table::New(env, std::move(groups.values)));
  }

  return result;
}

Napi::Value GroupBy::argmax(Napi::CallbackInfo const& info) {
  auto args   = _get_basic_args(info);
  auto values = args.first;
  auto mr     = args.second;
  return _single_aggregation(
    [&]() { return cudf::make_argmax_aggregation<cudf::groupby_aggregation>(); }, values, mr, info);
}

Napi::Value GroupBy::argmin(Napi::CallbackInfo const& info) {
  auto args   = _get_basic_args(info);
  auto values = args.first;
  auto mr     = args.second;
  return _single_aggregation(
    [&]() { return cudf::make_argmin_aggregation<cudf::groupby_aggregation>(); }, values, mr, info);
}

Napi::Value GroupBy::count(Napi::CallbackInfo const& info) {
  auto args   = _get_basic_args(info);
  auto values = args.first;
  auto mr     = args.second;
  return _single_aggregation(
    [&]() { return cudf::make_count_aggregation<cudf::groupby_aggregation>(); }, values, mr, info);
}

Napi::Value GroupBy::max(Napi::CallbackInfo const& info) {
  auto args   = _get_basic_args(info);
  auto values = args.first;
  auto mr     = args.second;
  return _single_aggregation(
    [&]() { return cudf::make_max_aggregation<cudf::groupby_aggregation>(); }, values, mr, info);
}

Napi::Value GroupBy::mean(Napi::CallbackInfo const& info) {
  auto args   = _get_basic_args(info);
  auto values = args.first;
  auto mr     = args.second;
  return _single_aggregation(
    [&]() { return cudf::make_mean_aggregation<cudf::groupby_aggregation>(); }, values, mr, info);
}

Napi::Value GroupBy::median(Napi::CallbackInfo const& info) {
  auto args   = _get_basic_args(info);
  auto values = args.first;
  auto mr     = args.second;
  return _single_aggregation(
    [&]() { return cudf::make_median_aggregation<cudf::groupby_aggregation>(); }, values, mr, info);
}

Napi::Value GroupBy::min(Napi::CallbackInfo const& info) {
  auto args   = _get_basic_args(info);
  auto values = args.first;
  auto mr     = args.second;
  return _single_aggregation(
    [&]() { return cudf::make_min_aggregation<cudf::groupby_aggregation>(); }, values, mr, info);
}

Napi::Value GroupBy::nth(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};

  cudf::size_type n = args[0];

  auto values = args[1];
  NODE_CUDA_EXPECT(Table::IsInstance(values),
                   "aggregation expects options to have a 'values' table");
  nv::Table* values_table = Table::Unwrap(values.ToObject());

  auto mr = MemoryResource::IsInstance(info[2]) ? *MemoryResource::Unwrap(info[2].ToObject())
                                                : rmm::mr::get_current_device_resource();

  return _single_aggregation(
    [&]() { return cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(n); },
    values_table,
    mr,
    info);
}

Napi::Value GroupBy::nunique(Napi::CallbackInfo const& info) {
  auto args   = _get_basic_args(info);
  auto values = args.first;
  auto mr     = args.second;
  return _single_aggregation(
    [&]() { return cudf::make_nunique_aggregation<cudf::groupby_aggregation>(); },
    values,
    mr,
    info);
}

Napi::Value GroupBy::std(Napi::CallbackInfo const& info) {
  auto args   = _get_basic_args(info);
  auto values = args.first;
  auto mr     = args.second;
  return _single_aggregation(
    [&]() { return cudf::make_std_aggregation<cudf::groupby_aggregation>(); }, values, mr, info);
}

Napi::Value GroupBy::sum(Napi::CallbackInfo const& info) {
  auto args   = _get_basic_args(info);
  auto values = args.first;
  auto mr     = args.second;
  return _single_aggregation(
    [&]() { return cudf::make_sum_aggregation<cudf::groupby_aggregation>(); }, values, mr, info);
}

Napi::Value GroupBy::var(Napi::CallbackInfo const& info) {
  auto args   = _get_basic_args(info);
  auto values = args.first;
  auto mr     = args.second;
  return _single_aggregation(
    [&]() { return cudf::make_variance_aggregation<cudf::groupby_aggregation>(); },
    values,
    mr,
    info);
}

Napi::Value GroupBy::quantile(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};

  double q = args[0];
  std::vector<double> qs{q};

  auto values = args[1];
  NODE_CUDA_EXPECT(Table::IsInstance(values),
                   "GroupBy quantile_agg expects options to have a 'values' table");
  nv::Table* values_table = Table::Unwrap(values.ToObject());

  cudf::interpolation interpolation = args[2];

  auto mr = MemoryResource::IsInstance(info[3]) ? *MemoryResource::Unwrap(info[3].ToObject())
                                                : rmm::mr::get_current_device_resource();

  return _single_aggregation(
    [&]() { return cudf::make_quantile_aggregation<cudf::groupby_aggregation>(qs, interpolation); },
    values_table,
    mr,
    info);
}

std::pair<nv::Table*, rmm::mr::device_memory_resource*> GroupBy::_get_basic_args(
  Napi::CallbackInfo const& info) {
  CallbackArgs args{info};

  auto values = args[0];
  NODE_CUDA_EXPECT(Table::IsInstance(values), "aggregation expects to have a 'values' table");

  rmm::mr::device_memory_resource* mr = args[1];

  return std::pair<Table*, rmm::mr::device_memory_resource*>(Table::Unwrap(values.ToObject()), mr);
}

template <typename MakeAggregation>
Napi::Value GroupBy::_single_aggregation(MakeAggregation const& make_aggregation,
                                         const nv::Table* const values_table,
                                         rmm::mr::device_memory_resource* const mr,
                                         Napi::CallbackInfo const& info) {
  auto env = info.Env();

  std::vector<cudf::groupby::aggregation_request> requests;

  for (cudf::size_type i = 0; i < values_table->num_columns(); ++i) {
    auto request   = cudf::groupby::aggregation_request();
    request.values = values_table->get_column(i).view();
    request.aggregations.push_back(std::move(make_aggregation()));
    requests.emplace_back(std::move(request));
  }

  std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::groupby::aggregation_result>> result;

  try {
    result = groupby_->aggregate(requests, mr);
  } catch (cudf::logic_error const& e) { NAPI_THROW(Napi::Error::New(env, e.what())); }

  auto result_keys = Table::New(env, std::move(result.first));
  auto result_cols = Napi::Array::New(env, result.second.size());

  for (size_t i = 0; i < result.second.size(); ++i) {
    result_cols.Set(i, Column::New(env, std::move(result.second[i].results[0]))->Value());
  }

  auto obj = Napi::Object::New(env);
  obj.Set("keys", result_keys);
  obj.Set("cols", result_cols);

  return obj;
}

}  // namespace nv
