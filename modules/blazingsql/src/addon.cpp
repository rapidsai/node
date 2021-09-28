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

#include "context.hpp"

#include <blazingsql/api.hpp>
#include <blazingsql/cache.hpp>
#include <blazingsql/graph.hpp>

#include <node_cudf/table.hpp>

#include <nv_node/utilities/args.hpp>

struct node_blazingsql : public nv::EnvLocalAddon, public Napi::Addon<node_blazingsql> {
  node_blazingsql(Napi::Env env, Napi::Object exports) : nv::EnvLocalAddon(env, exports) {
    DefineAddon(
      exports,
      {InstanceMethod("init", &node_blazingsql::InitAddon),
       InstanceMethod<&node_blazingsql::get_table_scan_info>("getTableScanInfo"),
       InstanceMethod<&node_blazingsql::run_generate_physical_graph>("runGeneratePhysicalGraph"),
       InstanceMethod<&node_blazingsql::parse_schema>("parseSchema"),
       InstanceValue("_cpp_exports", _cpp_exports.Value()),
       InstanceValue("Context", InitClass<nv::Context>(env, exports)),
       InstanceValue("CacheMachine", InitClass<nv::CacheMachine>(env, exports)),
       InstanceValue("ExecutionGraphWrapper", InitClass<nv::ExecutionGraph>(env, exports)),
       InstanceValue("UcpContext", InitClass<nv::UcpContext>(env, exports)),
       InstanceValue("ContextWrapper", InitClass<nv::ContextWrapper>(env, exports))});
  }

 private:
  Napi::Value get_table_scan_info(Napi::CallbackInfo const& info) {
    auto env            = info.Env();
    auto [names, steps] = nv::get_table_scan_info(info[0].ToString());

    Napi::Array table_names = Napi::Array::New(env, names.size());
    Napi::Array table_scans = Napi::Array::New(env, steps.size());
    for (std::size_t i = 0; i < names.size(); ++i) {
      table_names[i] = Napi::String::New(env, names[i]);
    }
    for (std::size_t i = 0; i < steps.size(); ++i) {
      table_scans[i] = Napi::String::New(env, steps[i]);
    }

    auto result = Napi::Array::New(env, 2);
    result.Set(0u, table_names);
    result.Set(1u, table_scans);

    return result;
  }

  Napi::Value run_generate_physical_graph(Napi::CallbackInfo const& info) {
    auto env = info.Env();
    nv::CallbackArgs args{info};

    uint32_t masterIndex                = args[0];
    std::vector<std::string> worker_ids = args[1];
    int32_t ctx_token                   = args[2];
    std::string query                   = args[3];

    return Napi::String::New(
      env, nv::run_generate_physical_graph(masterIndex, worker_ids, ctx_token, query));
  }

  Napi::Value parse_schema(Napi::CallbackInfo const& info) {
    auto env = info.Env();
    nv::CallbackArgs args{info};

    std::vector<std::string> input = args[0];
    std::string file_format        = args[1];
    // skip kwargs for now.
    // skip extraColumns for now.
    bool ignoreMissingPaths = args[4];

    return nv::parse_schema(env, input, file_format, ignoreMissingPaths);
  }
};

NODE_API_ADDON(node_blazingsql);
