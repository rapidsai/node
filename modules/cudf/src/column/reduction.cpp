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

#include <node_cudf/column.hpp>
#include <node_cudf/scalar.hpp>
#include <node_cudf/utilities/dtypes.hpp>
#include <node_cudf/utilities/napi_to_cpp.hpp>

#include <nv_node/utilities/args.hpp>
#include <nv_node/utilities/napi_to_cpp.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/reduction.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <memory>
#include <string>

namespace nv {

namespace {
cudf::data_type _compute_dtype(cudf::type_id id) {
  switch (id) {
    case cudf::type_id::INT8:
    case cudf::type_id::INT16:
    case cudf::type_id::INT32:
    case cudf::type_id::INT64: return cudf::data_type{cudf::type_id::INT64};
    case cudf::type_id::UINT8:
    case cudf::type_id::UINT16:
    case cudf::type_id::UINT32:
    case cudf::type_id::UINT64: return cudf::data_type{cudf::type_id::UINT64};
    default: return cudf::data_type{cudf::type_id::FLOAT64};
  }
}
}  // namespace

std::pair<Scalar::wrapper_t, Scalar::wrapper_t> Column::minmax(
  rmm::mr::device_memory_resource* mr) const {
  try {
    auto result = cudf::minmax(*this, mr);
    return {Scalar::New(Env(), std::move(result.first)),  //
            Scalar::New(Env(), std::move(result.second))};
  } catch (std::exception const& e) { NAPI_THROW(Napi::Error::New(Env(), e.what())); }
}

Napi::Value Column::min(Napi::CallbackInfo const& info) {
  return Napi::Value::From(
    info.Env(), minmax(NapiToCPP(info[0]).operator rmm::mr::device_memory_resource*()).first);
}

Napi::Value Column::max(Napi::CallbackInfo const& info) {
  return Napi::Value::From(
    info.Env(), minmax(NapiToCPP(info[0]).operator rmm::mr::device_memory_resource*()).second);
}

Napi::Value Column::minmax(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(),
                           minmax(NapiToCPP(info[0]).operator rmm::mr::device_memory_resource*()));
}

Scalar::wrapper_t Column::reduce(std::unique_ptr<cudf::aggregation> const& agg,
                                 cudf::data_type const& output_dtype,
                                 rmm::mr::device_memory_resource* mr) const {
  try {
    return Scalar::New(Env(), cudf::reduce(*this, agg, output_dtype, mr));
  } catch (std::exception const& e) { NAPI_THROW(Napi::Error::New(Env(), e.what())); }
}

Column::wrapper_t Column::scan(std::unique_ptr<cudf::aggregation> const& agg,
                               cudf::scan_type inclusive,
                               cudf::null_policy null_handling,
                               rmm::mr::device_memory_resource* mr) const {
  try {
    return Column::New(Env(), cudf::scan(*this, agg, inclusive, null_handling, mr));
  } catch (std::exception const& e) { NAPI_THROW(Napi::Error::New(Env(), e.what())); }
}

Scalar::wrapper_t Column::sum(rmm::mr::device_memory_resource* mr) const {
  return reduce(cudf::make_sum_aggregation(), _compute_dtype(this->type().id()), mr);
}

Napi::Value Column::sum(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(),
                           sum(NapiToCPP(info[0]).operator rmm::mr::device_memory_resource*()));
}

Scalar::wrapper_t Column::product(rmm::mr::device_memory_resource* mr) const {
  return reduce(cudf::make_product_aggregation(), _compute_dtype(this->type().id()), mr);
}

Napi::Value Column::product(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(),
                           product(NapiToCPP(info[0]).operator rmm::mr::device_memory_resource*()));
}

Scalar::wrapper_t Column::sum_of_squares(rmm::mr::device_memory_resource* mr) const {
  return reduce(cudf::make_sum_of_squares_aggregation(), _compute_dtype(this->type().id()), mr);
}

Napi::Value Column::sum_of_squares(Napi::CallbackInfo const& info) {
  return Napi::Value::From(
    info.Env(), sum_of_squares(NapiToCPP(info[0]).operator rmm::mr::device_memory_resource*()));
}

Napi::Value Column::any(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  return Napi::Value::From(
    info.Env(),
    reduce(cudf::make_any_aggregation(), cudf::data_type(cudf::type_id::BOOL8), args[0]));
}

Napi::Value Column::all(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  return Napi::Value::From(
    info.Env(),
    reduce(cudf::make_all_aggregation(), cudf::data_type(cudf::type_id::BOOL8), args[0]));
}

Scalar::wrapper_t Column::mean(rmm::mr::device_memory_resource* mr) const {
  return reduce(cudf::make_mean_aggregation(), cudf::data_type(cudf::type_id::FLOAT64), mr);
}

Napi::Value Column::mean(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(),
                           mean(NapiToCPP(info[0]).operator rmm::mr::device_memory_resource*()));
}

Scalar::wrapper_t Column::median(rmm::mr::device_memory_resource* mr) const {
  return reduce(cudf::make_median_aggregation(), cudf::data_type(cudf::type_id::FLOAT64), mr);
}

Napi::Value Column::median(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(),
                           median(NapiToCPP(info[0]).operator rmm::mr::device_memory_resource*()));
}

Scalar::wrapper_t Column::nunique(bool dropna, rmm::mr::device_memory_resource* mr) const {
  cudf::null_policy null_policy =
    (dropna == true) ? cudf::null_policy::EXCLUDE : cudf::null_policy::INCLUDE;
  return reduce(cudf::make_nunique_aggregation(null_policy),
                cudf::data_type{cudf::type_to_id<cudf::size_type>()},
                mr);
}

Napi::Value Column::nunique(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  return Napi::Value::From(info.Env(), nunique(args[0], args[1]));
}

Scalar::wrapper_t Column::variance(cudf::size_type ddof,
                                   rmm::mr::device_memory_resource* mr) const {
  return reduce(cudf::make_variance_aggregation(ddof), cudf::data_type(cudf::type_id::FLOAT64), mr);
}

Napi::Value Column::variance(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  return Napi::Value::From(info.Env(), variance(args[0], args[1]));
}

Scalar::wrapper_t Column::std(cudf::size_type ddof, rmm::mr::device_memory_resource* mr) const {
  return reduce(cudf::make_std_aggregation(ddof), cudf::data_type(cudf::type_id::FLOAT64), mr);
}

Napi::Value Column::std(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  return Napi::Value::From(info.Env(), std(args[0], args[1]));
}

Scalar::wrapper_t Column::quantile(double q,
                                   cudf::interpolation i,
                                   rmm::mr::device_memory_resource* mr) const {
  return reduce(
    cudf::make_quantile_aggregation({q}, i), cudf::data_type(cudf::type_id::FLOAT64), mr);
}

Napi::Value Column::quantile(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  return Napi::Value::From(info.Env(), quantile(args[0], args[1], args[2]));
}

Column::wrapper_t Column::cummax(rmm::mr::device_memory_resource* mr) const {
  return scan(cudf::make_max_aggregation(),
              // following cudf, scan type and null policy always use these values
              cudf::scan_type::INCLUSIVE,
              cudf::null_policy::EXCLUDE,
              mr);
}

Napi::Value Column::cummax(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(),
                           cummax(NapiToCPP(info[0]).operator rmm::mr::device_memory_resource*()));
}

Column::wrapper_t Column::cummin(rmm::mr::device_memory_resource* mr) const {
  return scan(cudf::make_min_aggregation(),
              // following cudf, scan type and null policy always use these values
              cudf::scan_type::INCLUSIVE,
              cudf::null_policy::EXCLUDE,
              mr);
}

Napi::Value Column::cummin(Napi::CallbackInfo const& info) {
  return cummin(NapiToCPP(info[0]).operator rmm::mr::device_memory_resource*())->Value();
}

Column::wrapper_t Column::cumprod(rmm::mr::device_memory_resource* mr) const {
  return scan(cudf::make_product_aggregation(),
              // following cudf, scan type and null policy always use these values
              cudf::scan_type::INCLUSIVE,
              cudf::null_policy::EXCLUDE,
              mr);
}

Napi::Value Column::cumprod(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(),
                           cumprod(NapiToCPP(info[0]).operator rmm::mr::device_memory_resource*()));
}

Column::wrapper_t Column::cumsum(rmm::mr::device_memory_resource* mr) const {
  return scan(cudf::make_sum_aggregation(),
              // following cudf, scan type and null policy always use these values
              cudf::scan_type::INCLUSIVE,
              cudf::null_policy::EXCLUDE,
              mr);
}

Napi::Value Column::cumsum(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(),
                           cumsum(NapiToCPP(info[0]).operator rmm::mr::device_memory_resource*()));
}

}  // namespace nv
