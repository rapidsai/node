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

#include <blazingsql/api.hpp>
#include <blazingsql/cache.hpp>
#include <blazingsql/contextwrapper.hpp>
#include <blazingsql/graph.hpp>
#include "context.hpp"

struct node_blazingsql : public nv::EnvLocalAddon, public Napi::Addon<node_blazingsql> {
  node_blazingsql(Napi::Env env, Napi::Object exports) : nv::EnvLocalAddon(env, exports) {
    DefineAddon(
      exports,
      {InstanceMethod("init", &node_blazingsql::InitAddon),
       InstanceMethod<&node_blazingsql::run_generate_graph>("runGenerateGraph"),
       InstanceMethod<&node_blazingsql::get_table_scan_info>("getTableScanInfo"),
       InstanceMethod<&node_blazingsql::start_execute_graph>("startExecuteGraph"),
       InstanceMethod<&node_blazingsql::get_execute_graph_result>("getExecuteGraphResult"),
       InstanceValue("_cpp_exports", _cpp_exports.Value()),
       InstanceValue("Context", InitClass<nv::Context>(env, exports)),
       InstanceValue("CacheMachine", InitClass<nv::CacheMachine>(env, exports)),
       InstanceValue("ExecutionGraph", InitClass<nv::ExecutionGraph>(env, exports)),
       InstanceValue("ContextWrapper", InitClass<nv::ContextWrapper>(env, exports))});
  }

 private:
  Napi::Value run_generate_graph(Napi::CallbackInfo const& info) {
    return nv::run_generate_graph(info);
  }

  Napi::Value get_table_scan_info(Napi::CallbackInfo const& info) {
    return nv::get_table_scan_info(info);
  }

  void start_execute_graph(Napi::CallbackInfo const& info) { nv::start_execute_graph(info); }

  Napi::Value get_execute_graph_result(Napi::CallbackInfo const& info) {
    return nv::get_execute_graph_result(info);
  }
};

NODE_API_ADDON(node_blazingsql);
