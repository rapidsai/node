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

#include <blazingsql/cache.hpp>
#include <blazingsql/contextwrapper.hpp>
#include <blazingsql/graph.hpp>
#include "context.hpp"

#include <nv_node/addon.hpp>

struct node_blazingsql : public nv::EnvLocalAddon, public Napi::Addon<node_blazingsql> {
  node_blazingsql(Napi::Env env, Napi::Object exports) : nv::EnvLocalAddon(env, exports) {
    DefineAddon(exports,
                {InstanceMethod("init", &node_blazingsql::InitAddon),
                 InstanceValue("_cpp_exports", _cpp_exports.Value()),
                 InstanceValue("Context", InitClass<nv::Context>(env, exports)),
                 InstanceValue("CacheMachine", InitClass<nv::CacheMachine>(env, exports)),
                 InstanceValue("ExecutionGraph", InitClass<nv::ExecutionGraph>(env, exports)),
                 InstanceValue("ContextWrapper", InitClass<nv::ContextWrapper>(env, exports))});
  }
};

NODE_API_ADDON(node_blazingsql);
