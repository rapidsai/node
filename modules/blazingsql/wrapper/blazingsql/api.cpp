// // Copyright (c) 2021, NVIDIA CORPORATION.
// //
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// //     http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // See the License for the specific language governing permissions and
// // limitations under the License.

#include "api.hpp"

#include <engine/engine.h>
#include <engine/initialize.h>

namespace nv {

ContextWrapper::wrapper_t initialize(Napi::Env const& env, NapiToCPP::Object const& props) {
  uint16_t ral_id                               = props.Get("ralId");
  std::string worker_id                         = props.Get("workerId");
  std::string network_iface_name                = props.Get("network_iface_name");
  int32_t ral_communication_port                = props.Get("ralCommunicationPort");
  std::vector<NodeMetaDataUCP> workers_ucp_info = props.Get("workersUcpInfo");
  bool single_node                              = props.Get("singleNode");
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

  auto init_result = ::initialize(ral_id,
                                  worker_id,
                                  network_iface_name,
                                  ral_communication_port,
                                  workers_ucp_info,
                                  single_node,
                                  config_options,
                                  allocation_mode,
                                  initial_pool_size,
                                  maximum_pool_size,
                                  enable_logging);
  return ContextWrapper::New(env, init_result);
}

Napi::Value get_table_scan_info(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};

  std::string logical_plan = args[0];
  auto table_scan_info     = ::getTableScanInfo(logical_plan);

  Napi::Array table_names = Napi::Array::New(env, table_scan_info.table_names.size());
  Napi::Array table_scans = Napi::Array::New(env, table_scan_info.relational_algebra_steps.size());
  for (int i = 0; i < table_scan_info.table_names.size(); ++i) {
    table_names[i] = Napi::String::New(env, table_scan_info.table_names[i]);
  }

  for (int i = 0; i < table_scan_info.relational_algebra_steps.size(); ++i) {
    table_scans[i] = Napi::String::New(env, table_scan_info.relational_algebra_steps[i]);
  }

  auto result = Napi::Array::New(env, 2);
  result.Set(0u, table_names);
  result.Set(1u, table_scans);

  return result;
}

ExecutionGraph::wrapper_t run_generate_graph(Napi::CallbackInfo const& info) {
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

  auto result = ::runGenerateGraph(masterIndex,
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
  return ExecutionGraph::New(env, result);
}

void start_execute_graph(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};

  ExecutionGraph::wrapper_t execution_graph = args[0];
  int32_t ctx_token                         = args[1];

  ::startExecuteGraph(execution_graph->graph(), ctx_token);
}

std::tuple<std::vector<std::string>, std::vector<std::unique_ptr<cudf::table>>>
get_execute_graph_result(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};

  ExecutionGraph::wrapper_t execution_graph = args[0];
  int32_t ctx_token                         = args[1];

  auto bsql_result = std::move(::getExecuteGraphResult(execution_graph->graph(), ctx_token));
  return {std::move(bsql_result->names), std::move(bsql_result->cudfTables)};
}

}  // namespace nv
