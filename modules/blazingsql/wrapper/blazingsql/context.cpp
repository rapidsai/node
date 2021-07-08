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
#include "cache.hpp"
#include "graph.hpp"

#include <node_cudf/table.hpp>
#include <nv_node/utilities/args.hpp>

#include <engine/engine.h>
#include <engine/initialize.h>
#include <io/io.h>

namespace nv {

Napi::Function Context::Init(Napi::Env env, Napi::Object exports) {
  return DefineClass(env,
                     "Context",
                     {
                       InstanceMethod<&Context::sql>("sql"),
                       InstanceAccessor<&Context::port>("port"),
                     });
}

Context::wrapper_t Context::New(Napi::Env const& env) {
  return EnvLocalObjectWrap<Context>::New(env);
}

Context::Context(Napi::CallbackInfo const& info) : EnvLocalObjectWrap<Context>(info) {
  auto env = info.Env();

  NapiToCPP::Object props                       = info[0];
  uint16_t ralId                                = props.Get("ralId");
  std::string worker_id                         = props.Get("workerId");
  std::string network_iface_name                = props.Get("network_iface_name");
  int32_t ralCommunicationPort                  = props.Get("ralCommunicationPort");
  std::vector<NodeMetaDataUCP> workers_ucp_info = props.Get("workersUcpInfo");
  bool singleNode                               = props.Get("singleNode");
  std::string allocation_mode                   = props.Get("allocationMode");
  std::size_t initial_pool_size                 = props.Get("initialPoolSize");
  std::size_t maximum_pool_size                 = props.Get("maximumPoolSize");
  bool enable_logging                           = props.Get("enableLogging");

  auto config_options = [&] {
    std::map<std::string, std::string> config{};
    auto prop = props.Get("configOptions");
    if (prop.IsObject() and not prop.IsNull()) {
      auto opts = prop.As<Napi::Object>();
      auto keys = opts.GetPropertyNames();
      for (auto i = 0u; i < keys.Length(); ++i) {
        Napi::HandleScope scope(env);
        auto name    = keys.Get(i).ToString();
        config[name] = opts.Get(name).ToString();
      }
    }
    return config;
  }();

  auto init_result = ::initialize(ralId,
                                  worker_id,
                                  network_iface_name,
                                  ralCommunicationPort,
                                  workers_ucp_info,
                                  singleNode,
                                  config_options,
                                  allocation_mode,
                                  initial_pool_size,
                                  maximum_pool_size,
                                  enable_logging);
  auto& caches     = init_result.first;
  _port            = init_result.second;
  _transport_out   = Napi::Persistent(CacheMachine::New(env, caches.first));
  _transport_in    = Napi::Persistent(CacheMachine::New(env, caches.second));
}

// TODO: These could be moved into their own methods, for now let's just chain call them.
void Context::sql(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};

  uint32_t masterIndex                                      = args[0];
  std::vector<std::string> worker_ids                       = args[1];
  std::vector<std::string> table_names                      = args[2];
  std::vector<std::string> table_scans                      = args[3];
  std::vector<Napi::Object> dataframes                      = args[4];
  std::vector<std::vector<std::string>> table_schema_keys   = args[5];
  std::vector<std::vector<std::string>> table_schema_values = args[6];
  std::vector<std::vector<std::string>> files_all           = args[7];
  std::vector<int> file_types                               = args[8];
  int32_t ctx_token                                         = args[9];
  std::string query                                         = args[10];
  // std::vector<std::vector<std::map<std::string, std::string>>> uri_values = args[11];
  std::string sql               = args[13];
  std::string current_timestamp = args[14];

  std::vector<TableSchema> schemas{{}};
  for (int i = 0; i < dataframes.size(); ++i) {
    std::vector<std::string> names;
    auto dfNames = dataframes[i].Get("names").As<Napi::Array>();
    for (size_t j = 0; j < dfNames.Length(); ++i) { names[j] = dfNames.Get(i).ToString(); }

    Table::wrapper_t table = dataframes[i].Get("asTable").As<Napi::Function>().Call({}).ToObject();
    schemas[0].blazingTableViews.push_back({table->view(), names});
  }

  auto config_options = [&] {
    std::map<std::string, std::string> config{};
    auto prop = args[12];
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

  std::cout << masterIndex << std::endl;
  std::cout << worker_ids.size() << std::endl;
  std::cout << table_names.size() << std::endl;
  std::cout << table_scans.size() << std::endl;
  std::cout << table_schema_keys.size() << std::endl;
  std::cout << table_schema_values.size() << std::endl;
  std::cout << files_all.size() << std::endl;
  std::cout << file_types.size() << std::endl;
  std::cout << ctx_token << std::endl;
  std::cout << query << std::endl;
  std::cout << sql << std::endl;
  std::cout << current_timestamp << std::endl;

  // auto result = ::runGenerateGraph(masterIndex,
  //                                  worker_ids,
  //                                  table_names,
  //                                  table_scans,
  //                                  {},
  //                                  table_schema_keys,
  //                                  table_schema_values,
  //                                  files_all,
  //                                  file_types,
  //                                  ctx_token,
  //                                  query,
  //                                  {},
  //                                  config_options,
  //                                  sql,
  //                                  current_timestamp);

  // // ::startExecuteGraph(result, ctx_token);
  // // auto finalResult = ::getExecuteGraphResult(result, ctxToken);
}

}  // namespace nv
