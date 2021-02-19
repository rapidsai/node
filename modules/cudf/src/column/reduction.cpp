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

std::pair<ObjectUnwrap<Scalar>, ObjectUnwrap<Scalar>> Column::minmax() const {
  auto result = cudf::minmax(*this);
  return {Scalar::New(std::move(result.first)),  //
          Scalar::New(std::move(result.second))};
}

Napi::Value Column::min(Napi::CallbackInfo const& info) { return minmax().first; }

Napi::Value Column::max(Napi::CallbackInfo const& info) { return minmax().second; }

ObjectUnwrap<Scalar> Column::reduce(std::unique_ptr<cudf::aggregation> const& agg,
                                    cudf::data_type const& output_dtype,
                                    rmm::mr::device_memory_resource* mr) const {
  return Scalar::New(cudf::reduce(*this, agg, output_dtype, mr));
}

ObjectUnwrap<Scalar> Column::sum(rmm::mr::device_memory_resource* mr) const {
  return reduce(cudf::make_sum_aggregation(), _compute_dtype(this->type().id()), mr);
}

Napi::Value Column::sum(Napi::CallbackInfo const& info) {
  return sum(NapiToCPP(info[0]).operator rmm::mr::device_memory_resource*());
}

ObjectUnwrap<Scalar> Column::product(rmm::mr::device_memory_resource* mr) const {
  return reduce(cudf::make_product_aggregation(), _compute_dtype(this->type().id()), mr);
}

Napi::Value Column::product(Napi::CallbackInfo const& info) {
  return product(NapiToCPP(info[0]).operator rmm::mr::device_memory_resource*());
}

ObjectUnwrap<Scalar> Column::sum_of_squares(rmm::mr::device_memory_resource* mr) const {
  return reduce(cudf::make_sum_of_squares_aggregation(), _compute_dtype(this->type().id()), mr);
}

Napi::Value Column::sum_of_squares(Napi::CallbackInfo const& info) {
  return sum_of_squares(NapiToCPP(info[0]).operator rmm::mr::device_memory_resource*());
}

Napi::Value Column::any(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  return reduce(cudf::make_any_aggregation(), cudf::data_type(cudf::type_id::BOOL8), args[0]);
}

Napi::Value Column::all(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  return reduce(cudf::make_all_aggregation(), cudf::data_type(cudf::type_id::BOOL8), args[0]);
}

ObjectUnwrap<Scalar> Column::mean(rmm::mr::device_memory_resource* mr) const {
  return reduce(cudf::make_mean_aggregation(), cudf::data_type(cudf::type_id::FLOAT64), mr);
}

Napi::Value Column::mean(Napi::CallbackInfo const& info) {
  return mean(NapiToCPP(info[0]).operator rmm::mr::device_memory_resource*());
}

ObjectUnwrap<Scalar> Column::median(rmm::mr::device_memory_resource* mr) const {
  return reduce(cudf::make_median_aggregation(), cudf::data_type(cudf::type_id::FLOAT64), mr);
}

Napi::Value Column::median(Napi::CallbackInfo const& info) {
  return median(NapiToCPP(info[0]).operator rmm::mr::device_memory_resource*());
}

ObjectUnwrap<Scalar> Column::nunique(bool skipna, rmm::mr::device_memory_resource* mr) const {
  cudf::null_policy null_policy =
    (skipna == true) ? cudf::null_policy::EXCLUDE : cudf::null_policy::INCLUDE;
  return reduce(cudf::make_nunique_aggregation(null_policy),
                cudf::data_type{cudf::type_to_id<cudf::size_type>()},
                mr);
}

Napi::Value Column::nunique(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  return nunique(args[0], args[1]);
}

ObjectUnwrap<Scalar> Column::variance(cudf::size_type ddof,
                                      rmm::mr::device_memory_resource* mr) const {
  return reduce(cudf::make_variance_aggregation(ddof), cudf::data_type(cudf::type_id::FLOAT64), mr);
}

Napi::Value Column::variance(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  return variance(args[0], args[1]);
}

ObjectUnwrap<Scalar> Column::std(cudf::size_type ddof, rmm::mr::device_memory_resource* mr) const {
  return reduce(cudf::make_std_aggregation(ddof), cudf::data_type(cudf::type_id::FLOAT64), mr);
}

Napi::Value Column::std(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  return std(args[0], args[1]);
}

ObjectUnwrap<Scalar> Column::quantile(double q,
                                      cudf::interpolation i,
                                      rmm::mr::device_memory_resource* mr) const {
  return reduce(
    cudf::make_quantile_aggregation({q}, i), cudf::data_type(cudf::type_id::FLOAT64), mr);
}

Napi::Value Column::quantile(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  return quantile(args[0], args[1], args[2]);
}

}  // namespace nv
