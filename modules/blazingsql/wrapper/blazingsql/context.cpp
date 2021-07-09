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
                       InstanceMethod<&Context::get_table_scan_info>("getTableScanInfo"),
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

  uint32_t masterIndex                  = args[0];
  std::vector<std::string> worker_ids   = args[1];
  std::vector<Napi::Object> data_frames = args[2];
  std::vector<Napi::Object> tables      = args[3];
  std::vector<std::string> table_names  = args[4];
  std::vector<std::string> table_scans  = args[5];
  int32_t ctx_token                     = args[6];
  std::string query                     = args[8];
  std::string sql                       = args[9];
  std::string current_timestamp         = args[10];

  std::vector<TableSchema> schemas{{}};
  for (int i = 0; i < data_frames.size(); ++i) {
    auto dfNames                   = data_frames[i].Get("names").As<Napi::Array>();
    std::vector<std::string> names = std::vector<std::string>(dfNames.Length());
    for (size_t j = 0; j < dfNames.Length(); ++j) { names[j] = dfNames.Get(i).ToString(); }

    Table::wrapper_t table = tables[i];
    schemas[0].blazingTableViews.push_back({table->view(), names});
  }

  auto config_options = [&] {
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

  // auto result = ::runGenerateGraph(masterIndex,
  //                                  worker_ids,
  //                                  table_names,
  //                                  table_scans,
  //                                  {},
  //                                  {},
  //                                  {},
  //                                  {},
  //                                  {},
  //                                  ctx_token,
  //                                  query,
  //                                  {},
  //                                  config_options,
  //                                  sql,
  //                                  current_timestamp);
  // // ::startExecuteGraph(result, ctx_token);
  // // auto finalResult = ::getExecuteGraphResult(result, ctxToken);
}

Napi::Value Context::get_table_scan_info(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};

  auto table_scan_info    = ::getTableScanInfo(args[0]);
  Napi::Array table_names = Napi::Array::New(info.Env(), table_scan_info.table_names.size());
  Napi::Array table_scans =
    Napi::Array::New(info.Env(), table_scan_info.relational_algebra_steps.size());

  for (int i = 0; i < table_scan_info.table_names.size(); ++i) {
    table_names[i] = Napi::String::New(info.Env(), table_scan_info.table_names[i]);
  }

  for (int i = 0; i < table_scan_info.relational_algebra_steps.size(); ++i) {
    table_scans[i] = Napi::String::New(info.Env(), table_scan_info.relational_algebra_steps[i]);
  }

  auto result = Napi::Array::New(info.Env(), 2);
  result.Set(0u, table_names);
  result.Set(1u, table_scans);

  return result;
}

}  // namespace nv
