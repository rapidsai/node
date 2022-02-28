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

#include <node_cudf/groupby.hpp>
#include <node_cudf/table.hpp>
#include <node_cudf/utilities/error.hpp>
#include <node_cudf/utilities/napi_to_cpp.hpp>

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
                       InstanceMethod<&GroupBy::collect_list>("_collect_list"),
                       InstanceMethod<&GroupBy::collect_set>("_collect_set"),
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
  auto [values, mr] = _get_basic_args(info);
  return _single_aggregation(
    info, values, mr, []() { return cudf::make_argmax_aggregation<cudf::groupby_aggregation>(); });
}

Napi::Value GroupBy::argmin(Napi::CallbackInfo const& info) {
  auto [values, mr] = _get_basic_args(info);
  return _single_aggregation(
    info, values, mr, []() { return cudf::make_argmin_aggregation<cudf::groupby_aggregation>(); });
}

Napi::Value GroupBy::count(Napi::CallbackInfo const& info) {
  auto [values, mr] = _get_basic_args(info);
  return _single_aggregation(
    info, values, mr, []() { return cudf::make_count_aggregation<cudf::groupby_aggregation>(); });
}

Napi::Value GroupBy::max(Napi::CallbackInfo const& info) {
  auto [values, mr] = _get_basic_args(info);
  return _single_aggregation(
    info, values, mr, []() { return cudf::make_max_aggregation<cudf::groupby_aggregation>(); });
}

Napi::Value GroupBy::mean(Napi::CallbackInfo const& info) {
  auto [values, mr] = _get_basic_args(info);
  return _single_aggregation(
    info, values, mr, []() { return cudf::make_mean_aggregation<cudf::groupby_aggregation>(); });
}

Napi::Value GroupBy::median(Napi::CallbackInfo const& info) {
  auto [values, mr] = _get_basic_args(info);
  return _single_aggregation(
    info, values, mr, []() { return cudf::make_median_aggregation<cudf::groupby_aggregation>(); });
}

Napi::Value GroupBy::min(Napi::CallbackInfo const& info) {
  auto [values, mr] = _get_basic_args(info);
  return _single_aggregation(
    info, values, mr, []() { return cudf::make_min_aggregation<cudf::groupby_aggregation>(); });
}

Napi::Value GroupBy::nth(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  auto [values, mr] = _get_basic_args(info);
  cudf::size_type n = args[2];
  auto include_nulls =
    info[3].ToBoolean() ? cudf::null_policy::INCLUDE : cudf::null_policy::EXCLUDE;
  return _single_aggregation(info, values, mr, [&]() {
    return cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(n, include_nulls);
  });
}

Napi::Value GroupBy::nunique(Napi::CallbackInfo const& info) {
  auto [values, mr] = _get_basic_args(info);
  auto include_nulls =
    info[3].ToBoolean() ? cudf::null_policy::INCLUDE : cudf::null_policy::EXCLUDE;
  return _single_aggregation(info, values, mr, [&]() {
    return cudf::make_nunique_aggregation<cudf::groupby_aggregation>(include_nulls);
  });
}

Napi::Value GroupBy::std(Napi::CallbackInfo const& info) {
  auto [values, mr]    = _get_basic_args(info);
  cudf::size_type ddof = info[3].IsNumber() ? info[3].ToNumber() : 1;
  return _single_aggregation(info, values, mr, [&]() {
    return cudf::make_std_aggregation<cudf::groupby_aggregation>(ddof);
  });
}

Napi::Value GroupBy::sum(Napi::CallbackInfo const& info) {
  auto [values, mr] = _get_basic_args(info);
  return _single_aggregation(
    info, values, mr, []() { return cudf::make_sum_aggregation<cudf::groupby_aggregation>(); });
}

Napi::Value GroupBy::var(Napi::CallbackInfo const& info) {
  auto [values, mr]    = _get_basic_args(info);
  cudf::size_type ddof = info[3].IsNumber() ? info[3].ToNumber() : 1;
  return _single_aggregation(info, values, mr, [&]() {
    return cudf::make_variance_aggregation<cudf::groupby_aggregation>(ddof);
  });
}

Napi::Value GroupBy::quantile(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  auto [values, mr] = _get_basic_args(info);
  std::vector<double> quantiles{args[2]};
  cudf::interpolation interp = args[3];
  return _single_aggregation(info, values, mr, [&]() {
    return cudf::make_quantile_aggregation<cudf::groupby_aggregation>(quantiles, interp);
  });
}

Napi::Value GroupBy::collect_list(Napi::CallbackInfo const& info) {
  auto [values, mr] = _get_basic_args(info);
  auto include_nulls =
    info[2].ToBoolean() ? cudf::null_policy::INCLUDE : cudf::null_policy::EXCLUDE;
  return _single_aggregation(info, values, mr, [&]() {
    return cudf::make_collect_list_aggregation<cudf::groupby_aggregation>(include_nulls);
  });
}

Napi::Value GroupBy::collect_set(Napi::CallbackInfo const& info) {
  auto [values, mr] = _get_basic_args(info);
  auto include_nulls =
    info[2].ToBoolean() ? cudf::null_policy::INCLUDE : cudf::null_policy::EXCLUDE;
  auto nulls_equal =
    info[3].ToBoolean() ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;
  auto nans_equal =
    info[4].ToBoolean() ? cudf::nan_equality::UNEQUAL : cudf::nan_equality::ALL_EQUAL;
  return _single_aggregation(info, values, mr, [&]() {
    return cudf::make_collect_set_aggregation<cudf::groupby_aggregation>(
      include_nulls, nulls_equal, nans_equal);
  });
}

std::pair<Table::wrapper_t, rmm::mr::device_memory_resource*> GroupBy::_get_basic_args(
  Napi::CallbackInfo const& info) {
  CallbackArgs args{info};

  auto values = args[0];
  NODE_CUDA_EXPECT(Table::IsInstance(values), "aggregation expects to have a 'values' table");

  return std::make_pair(values.ToObject(), args[1]);
}

template <typename MakeAggregation>
Napi::Value GroupBy::_single_aggregation(Napi::CallbackInfo const& info,
                                         Table::wrapper_t const& values_table,
                                         rmm::mr::device_memory_resource* const mr,
                                         MakeAggregation const& make_aggregation) {
  auto env = info.Env();

  std::vector<cudf::groupby::aggregation_request> requests;
  requests.reserve(values_table->num_columns());

  for (cudf::size_type i = 0; i < values_table->num_columns(); ++i) {
    auto request   = cudf::groupby::aggregation_request();
    request.values = values_table->get_column(i).view();
    request.aggregations.push_back(std::move(make_aggregation()));
    requests.push_back(std::move(request));
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
