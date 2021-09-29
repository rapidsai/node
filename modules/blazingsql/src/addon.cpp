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

#include <blazingsql_wrapper/api.hpp>
#include <blazingsql_wrapper/cache.hpp>
#include <blazingsql_wrapper/graph.hpp>

#include <node_cudf/table.hpp>

#include <nv_node/utilities/args.hpp>

struct rapidsai_sql : public nv::EnvLocalAddon, public Napi::Addon<rapidsai_sql> {
  rapidsai_sql(Napi::Env env, Napi::Object exports) : nv::EnvLocalAddon(env, exports) {
    DefineAddon(
      exports,
      {
        InstanceMethod("init", &rapidsai_sql::InitAddon),
        InstanceValue("_cpp_exports", _cpp_exports.Value()),
        InstanceValue("Context", InitClass<nv::blazingsql::Context>(env, exports)),
        InstanceValue("UcpContext", InitClass<nv::blazingsql::UcpContext>(env, exports)),
        InstanceValue("CacheMachine", InitClass<nv::blazingsql::CacheMachine>(env, exports)),
        InstanceValue("ExecutionGraph", InitClass<nv::blazingsql::ExecutionGraph>(env, exports)),
        InstanceMethod<&rapidsai_sql::parse_schema>("parseSchema"),
        InstanceMethod<&rapidsai_sql::get_table_scan_info>("getTableScanInfo"),
        InstanceMethod<&rapidsai_sql::run_generate_physical_graph>("runGeneratePhysicalGraph"),
      });
  }

 private:
  Napi::Value get_table_scan_info(Napi::CallbackInfo const& info) {
    auto env            = info.Env();
    auto [names, steps] = nv::blazingsql::get_table_scan_info(info[0].ToString());

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

    std::vector<std::string> worker_ids = args[0];
    int32_t ctx_token                   = args[1];
    std::string query                   = args[2];

    return Napi::String::New(
      env, nv::blazingsql::run_generate_physical_graph(0, worker_ids, ctx_token, query));
  }

  Napi::Value parse_schema(Napi::CallbackInfo const& info) {
    auto env = info.Env();
    nv::CallbackArgs args{info};

    std::vector<std::string> input = args[0];

    return nv::blazingsql::parse_schema(env, input, "csv", false);
  }
};

NODE_API_ADDON(rapidsai_sql);
