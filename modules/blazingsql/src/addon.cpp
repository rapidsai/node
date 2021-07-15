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
       InstanceMethod<&node_blazingsql::run_generate_graph>("runGenerateGraph"),
       InstanceMethod<&node_blazingsql::start_execute_graph>("startExecuteGraph"),
       InstanceMethod<&node_blazingsql::get_execute_graph_result>("getExecuteGraphResult"),
       InstanceValue("_cpp_exports", _cpp_exports.Value()),
       InstanceValue("Context", InitClass<nv::Context>(env, exports)),
       InstanceValue("CacheMachine", InitClass<nv::CacheMachine>(env, exports)),
       InstanceValue("ExecutionGraph", InitClass<nv::ExecutionGraph>(env, exports)),
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

  Napi::Value run_generate_graph(Napi::CallbackInfo const& info) {
    auto env = info.Env();
    nv::CallbackArgs args{info};

    uint32_t masterIndex                 = args[0];
    std::vector<std::string> worker_ids  = args[1];
    Napi::Array data_frames              = args[2];
    std::vector<std::string> table_names = args[3];
    std::vector<std::string> table_scans = args[4];
    int32_t ctx_token                    = args[5];
    std::string query                    = args[6];
    std::string sql                      = args[8];
    std::string current_timestamp        = args[9];
    auto config_options                  = [&] {
      std::map<std::string, std::string> config{};
      auto prop = args[7];
      if (prop.IsObject() and not prop.IsNull()) {
        auto opts = prop.As<Napi::Object>();
        auto keys = opts.GetPropertyNames();
        for (auto i = 0u; i < keys.Length(); ++i) {
          auto name    = keys.Get(i).ToString();
          config[name] = opts.Get(name).ToString();
        }
      }
      return config;
    }();

    std::vector<cudf::table_view> table_views;
    std::vector<std::vector<std::string>> column_names;

    table_views.reserve(data_frames.Length());
    column_names.reserve(data_frames.Length());

    auto tables = Napi::Array::New(env, data_frames.Length());

    for (std::size_t i = 0; i < data_frames.Length(); ++i) {
      nv::NapiToCPP::Object df       = data_frames.Get(i);
      std::vector<std::string> names = df.Get("names");
      Napi::Function asTable         = df.Get("asTable");
      nv::Table::wrapper_t table     = asTable.Call(df.val, {}).ToObject();

      tables.Set(i, table);
      table_views.push_back(*table);
      column_names.push_back(names);
    }

    return nv::run_generate_graph(info.Env(),
                                  masterIndex,
                                  worker_ids,
                                  table_views,
                                  column_names,
                                  table_names,
                                  table_scans,
                                  ctx_token,
                                  query,
                                  sql,
                                  current_timestamp,
                                  config_options);
  }

  void start_execute_graph(Napi::CallbackInfo const& info) {
    nv::start_execute_graph(info[0].ToObject(), info[1].ToNumber());
  }

  Napi::Value get_execute_graph_result(Napi::CallbackInfo const& info) {
    auto env = info.Env();
    auto [bsql_names, bsql_tables] =
      nv::get_execute_graph_result(info[0].ToObject(), info[1].ToNumber());

    auto result_names = Napi::Array::New(env, bsql_names.size());
    for (size_t i = 0; i < bsql_names.size(); ++i) {
      result_names.Set(i, Napi::String::New(env, bsql_names[i]));
    }

    auto result_tables = Napi::Array::New(env, bsql_tables.size());
    for (size_t i = 0; i < bsql_tables.size(); ++i) {
      result_tables.Set(i, nv::Table::New(env, std::move(bsql_tables[i])));
    }

    auto result = Napi::Object::New(env);
    result.Set("names", result_names);
    result.Set("tables", result_tables);
    return result;
  }
};

NODE_API_ADDON(node_blazingsql);
