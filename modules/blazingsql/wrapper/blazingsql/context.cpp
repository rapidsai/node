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
Napi::Value Context::sql(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};

  uint32_t masterIndex                 = args[0];
  std::vector<std::string> worker_ids  = args[1];
  Napi::Array data_frames              = args[2];
  std::vector<std::string> table_names = args[3];
  std::vector<std::string> table_scans = args[4];
  int32_t ctx_token                    = args[5];
  std::string query                    = args[6];
  std::string sql                      = args[8];
  std::string current_timestamp        = args[9];

  std::vector<TableSchema> table_schemas;
  std::vector<std::vector<std::string>> table_schema_cpp_arg_keys;
  std::vector<std::vector<std::string>> table_schema_cpp_arg_values;
  std::vector<std::vector<std::string>> files_all;
  std::vector<int> file_types;
  std::vector<std::vector<std::map<std::string, std::string>>> uri_values;

  auto cudf_tables = Napi::Array::New(env, data_frames.Length());

  table_schemas.reserve(data_frames.Length());
  table_schema_cpp_arg_keys.reserve(data_frames.Length());
  table_schema_cpp_arg_values.reserve(data_frames.Length());
  files_all.reserve(data_frames.Length());
  file_types.reserve(data_frames.Length());
  uri_values.reserve(data_frames.Length());

  for (std::size_t i = 0; i < data_frames.Length(); ++i) {
    NapiToCPP::Object df           = data_frames.Get(i);
    std::vector<std::string> names = df.Get("names");
    Napi::Function asTable         = df.Get("asTable");
    Table::wrapper_t table         = asTable.Call(df.val, {}).ToObject();

    cudf_tables.Set(i, table);

    std::vector<cudf::type_id> type_ids;
    for (auto const& col : table->view()) { type_ids.push_back(col.type().id()); }

    table_schemas.push_back({
      {{table->view(), names}},  // std::vector<ral::frame::BlazingTableView> blazingTableViews
      type_ids,                  // std::vector<cudf::type_id> types
      {},                        // std::vector<std::string> files
      {},                        // std::vector<std::string> datasource
      {table_names[i]},          // std::vector<std::string> names
      {},                        // std::vector<size_t> calcite_to_file_indices
      {},                        // std::vector<bool> in_file
      ral::io::DataType::CUDF,   // int data_type
      false,                     // bool has_header_csv = false
      {cudf::table_view{}, {}},  // ral::frame::BlazingTableView metadata
      {{0}},                     // std::vector<std::vector<int>> row_groups_ids
      nullptr                    // std::shared_ptr<arrow::Table> arrow_tabl
    });
    table_schema_cpp_arg_keys.push_back({});
    table_schema_cpp_arg_values.push_back({});
    files_all.push_back({});
    file_types.push_back(ral::io::DataType::CUDF);
    uri_values.push_back({});
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

  auto runGenerateGraphResult = ::runGenerateGraph(masterIndex,
                                                   worker_ids,
                                                   table_names,
                                                   table_scans,
                                                   table_schemas,
                                                   table_schema_cpp_arg_keys,
                                                   table_schema_cpp_arg_values,
                                                   files_all,
                                                   file_types,
                                                   ctx_token,
                                                   query,
                                                   uri_values,
                                                   config_options,
                                                   sql,
                                                   current_timestamp);

  ::startExecuteGraph(runGenerateGraphResult, ctx_token);

  auto bsql_result  = std::move(::getExecuteGraphResult(runGenerateGraphResult, ctx_token));
  auto& bsql_names  = bsql_result->names;
  auto& bsql_tables = bsql_result->cudfTables;

  auto result_names = Napi::Array::New(env, bsql_names.size());
  for (size_t i = 0; i < bsql_names.size(); ++i) {
    result_names.Set(i, Napi::String::New(env, bsql_names[i]));
  }

  auto result_tables = Napi::Array::New(env, bsql_tables.size());
  for (size_t i = 0; i < bsql_tables.size(); ++i) {
    result_tables.Set(i, Table::New(env, std::move(bsql_tables[i])));
  }

  auto result = Napi::Object::New(env);
  result.Set("names", result_names);
  result.Set("tables", result_tables);
  return result;
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
