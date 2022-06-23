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

#include "node_cudf/column.hpp"
#include "node_cudf/groupby.hpp"
#include "node_cudf/scalar.hpp"
#include "node_cudf/table.hpp"
#include "node_cudf/utilities/dtypes.hpp"

#include <nv_node/addon.hpp>
#include <nv_node/macros.hpp>

#include <napi.h>

struct rapidsai_cudf : public nv::EnvLocalAddon, public Napi::Addon<rapidsai_cudf> {
  rapidsai_cudf(Napi::Env const& env, Napi::Object exports) : EnvLocalAddon(env, exports) {
    DefineAddon(exports,
                {
                  InstanceMethod("init", &rapidsai_cudf::InitAddon),
                  InstanceValue("_cpp_exports", _cpp_exports.Value()),

                  InstanceMethod<&rapidsai_cudf::find_common_type>("findCommonType"),

                  InstanceValue("Column", InitClass<nv::Column>(env, exports)),
                  InstanceValue("Table", InitClass<nv::Table>(env, exports)),
                  InstanceValue("Scalar", InitClass<nv::Scalar>(env, exports)),
                  InstanceValue("GroupBy", InitClass<nv::GroupBy>(env, exports)),
                });
  }

 private:
  Napi::Value find_common_type(Napi::CallbackInfo const& info) {
    return nv::find_common_type(info);
  }
};

NODE_API_ADDON(rapidsai_cudf);
