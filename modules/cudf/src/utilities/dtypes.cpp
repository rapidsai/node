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

#include <node_cudf/types.hpp>
#include <node_cudf/utilities/dtypes.hpp>
#include <node_cudf/utilities/napi_to_cpp.hpp>

#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

namespace nv {

namespace {

template <typename _RHS>
struct get_common_type_t {
  template <
    typename LHS,
    typename RHS                                          = _RHS,
    typename std::enable_if_t<(std::is_convertible<LHS, RHS>::value && cudf::is_numeric<LHS>() &&
                               cudf::is_numeric<RHS>())>* = nullptr>
  cudf::type_id operator()(cudf::data_type const& lhs, cudf::data_type const& rhs) {
    return cudf::type_to_id<std::common_type_t<LHS, RHS>>();
  }
  template <
    typename LHS,
    typename RHS                                           = _RHS,
    typename std::enable_if_t<!(std::is_convertible<LHS, RHS>::value && cudf::is_numeric<LHS>() &&
                                cudf::is_numeric<RHS>())>* = nullptr>
  cudf::type_id operator()(cudf::data_type const& lhs, cudf::data_type const& rhs) {
    auto lhs_name = cudf::type_dispatcher(lhs, cudf::type_to_name{});
    auto rhs_name = cudf::type_dispatcher(rhs, cudf::type_to_name{});
    CUDF_FAIL("Cannot determine a logical common type between " + lhs_name + " and " + rhs_name);
  }
};

struct dispatch_get_common_type_t {
  template <typename RHS>
  cudf::type_id operator()(cudf::data_type const& lhs, cudf::data_type const& rhs) {
    return cudf::type_dispatcher(lhs, get_common_type_t<RHS>{}, lhs, rhs);
  }
};

}  // namespace

cudf::type_id get_common_type(cudf::data_type const& lhs, cudf::data_type const& rhs) {
  return cudf::type_dispatcher(rhs, dispatch_get_common_type_t{}, lhs, rhs);
}

Napi::Value find_common_type(CallbackArgs const& args) {
  return DataType::New(get_common_type(args[0], args[1]))->Value();
}

}  // namespace nv
