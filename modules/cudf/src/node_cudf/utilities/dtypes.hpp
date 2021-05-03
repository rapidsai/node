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

#pragma once

#include <nv_node/utilities/args.hpp>

#include <cudf/types.hpp>

namespace nv {

cudf::type_id get_common_type(cudf::data_type const& lhs, cudf::data_type const& rhs);

cudf::data_type arrow_to_cudf_type(Napi::Object const& arrow_type);

Napi::Object cudf_to_arrow_type(Napi::Env const& env, cudf::data_type const& cudf_type);

Napi::Object column_to_arrow_type(Napi::Env const& env, cudf::column_view const& column);

Napi::Value find_common_type(CallbackArgs const& args);

}  // namespace nv

namespace Napi {

template <>
inline Value Value::From(napi_env env, cudf::data_type const& type) {
  return nv::cudf_to_arrow_type(env, type);
}

}  // namespace Napi
