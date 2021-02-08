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
#include <node_cudf/utilities/error.hpp>
#include <node_cudf/utilities/napi_to_cpp.hpp>

#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

namespace nv {

namespace {

template <typename LHS, typename RHS, typename = void>
struct common_type_exists : std::false_type {};

template <typename LHS, typename RHS>
struct common_type_exists<LHS, RHS, std::void_t<std::common_type_t<LHS, RHS>>> : std::true_type {};

struct get_common_type_id {
  template <typename LHS,
            typename RHS,
            typename std::enable_if_t<common_type_exists<LHS, RHS>::value>* = nullptr>
  cudf::type_id operator()() {
    return cudf::type_to_id<std::common_type_t<LHS, RHS>>();
  }
  template <typename LHS,
            typename RHS,
            typename std::enable_if_t<not common_type_exists<LHS, RHS>::value>* = nullptr>
  cudf::type_id operator()() {
    CUDF_FAIL("Cannot determine a logical common type between " +
              cudf::type_to_name{}.operator()<LHS>() + " and " +
              cudf::type_to_name{}.operator()<RHS>());
  }
};

}  // namespace

cudf::type_id get_common_type(cudf::data_type const& lhs, cudf::data_type const& rhs) {
  return cudf::double_type_dispatcher(lhs, rhs, get_common_type_id{});
}

Napi::Value find_common_type(CallbackArgs const& args) {
  try {
    return DataType::New(get_common_type(args[0], args[1]))->Value();
  } catch (cudf::logic_error const& err) { NODE_CUDF_THROW(err.what()); }
}

}  // namespace nv
