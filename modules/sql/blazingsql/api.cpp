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

#include "blazingsql_wrapper/api.hpp"
#include <node_cudf/utilities/dtypes.hpp>
#include "blazingsql_wrapper/ucpcontext.hpp"
#include "cudf/types.hpp"

#include <engine/engine.h>
#include <engine/initialize.h>
#include <algorithm>
#include <cstdint>

namespace nv {
namespace blazingsql {

std::tuple<uint16_t,
           int32_t,
           std::vector<int32_t>,
           UcpContext::wrapper_t,
           std::shared_ptr<ral::cache::CacheMachine>,
           std::shared_ptr<ral::cache::CacheMachine>>
initialize(Napi::Env const& env, NapiToCPP::Object const& props) {
  auto config_options = [&] {
    std::map<std::string, std::string> config{};
    auto prop = props.Get("configOptions");
    if (!prop.IsNull() && prop.IsObject()) {
      auto opts = prop.As<Napi::Object>();
      auto keys = opts.GetPropertyNames();
      for (auto i = 0u; i < keys.Length(); ++i) {
        Napi::HandleScope scope(env);
        std::string name = keys.Get(i).ToString();
        config[name]     = opts.Get(name).ToString();
        if (config[name] == "true") {
          config[name] = "True";
        } else if (config[name] == "false") {
          config[name] = "False";
        }
      }
    }
    return config;
  }();

  std::vector<int32_t> worker_ids{};
  UcpContext::wrapper_t ucp_context{};
  std::vector<NodeMetaDataUCP> ucp_metadata{};

  if (UcpContext::IsInstance(props.Get("ucpContext"))) {
    ucp_context = props.Get("ucpContext").ToObject();
    if (props.Get("workersUcpInfo").IsArray()) {
      auto list = props.Get("workersUcpInfo").As<Napi::Array>();
      worker_ids.reserve(list.Length());
      ucp_metadata.reserve(list.Length());
      for (size_t i = 0; i < list.Length(); ++i) {
        NapiToCPP::Object worker = list.Get(i);
        worker_ids.push_back(worker.Get("id"));
        ucp_metadata.push_back({
          worker.Get("id").ToString(),    // std::string worker_id
          worker.Get("ip").ToString(),    // std::string ip
          0,                              // std::uintptr_t ep_handle
          0,                              // std::uintptr_t worker_handle
          *ucp_context,                   // std::uintptr_t context_handle
          worker.Get("port").ToNumber(),  // int32_t port
        });
      }
    }
  }

  uint16_t id      = props.Get("id");
  bool single_node = ucp_metadata.size() == 0;
  if (single_node) { worker_ids.push_back(id); }

  auto init_result = std::move(::initialize(id,
                                            std::to_string(id),
                                            props.Get("networkIfaceName"),
                                            props.Get("port"),
                                            ucp_metadata,
                                            single_node,
                                            config_options,
                                            props.Get("allocationMode"),
                                            props.Get("initialPoolSize"),
                                            props.Get("maximumPoolSize"),
                                            props.Get("enableLogging")));

  auto& caches        = init_result.first;
  auto& port          = init_result.second;
  auto& transport_in  = caches.second;
  auto& transport_out = caches.first;

  return std::make_tuple(id,
                         port,
                         std::move(worker_ids),
                         std::move(ucp_context),
                         std::move(transport_in),
                         std::move(transport_out));
}

std::tuple<std::vector<std::string>, std::vector<std::string>> get_table_scan_info(
  std::string const& logical_plan) {
  auto table_scan_info = ::getTableScanInfo(logical_plan);
  return std::make_tuple(std::move(table_scan_info.table_names),
                         std::move(table_scan_info.relational_algebra_steps));
}

ExecutionGraph::wrapper_t run_generate_graph(
  Napi::Env const& env,
  Wrapper<Context> const& context,
  uint32_t const& masterIndex,
  std::vector<std::string> const& worker_ids,
  std::vector<cudf::table_view> const& table_views,
  Napi::Array const& schemas,
  std::vector<std::vector<std::string>> const& column_names,
  std::vector<std::string> const& table_names,
  std::vector<std::string> const& table_scans,
  int32_t const& ctx_token,
  std::string const& query,
  std::string const& sql,
  std::string const& current_timestamp,
  std::map<std::string, std::string> const& config_options) {
  std::vector<TableSchema> table_schemas;
  std::vector<std::vector<std::string>> table_schema_cpp_arg_keys;
  std::vector<std::vector<std::string>> table_schema_cpp_arg_values;
  std::vector<std::vector<std::string>> files_all;
  std::vector<int> file_types;
  std::vector<std::vector<std::map<std::string, std::string>>> uri_values;

  table_schemas.reserve(table_views.size() + schemas.Length());
  table_schema_cpp_arg_keys.reserve(table_views.size());
  table_schema_cpp_arg_values.reserve(table_views.size());
  files_all.reserve(table_views.size());
  file_types.reserve(table_views.size());
  uri_values.reserve(table_views.size());

  for (std::size_t i = 0; i < table_views.size(); ++i) {
    auto table = table_views[i];
    auto names = column_names[i];

    std::vector<cudf::type_id> type_ids;
    type_ids.reserve(table.num_columns());
    for (auto const& col : table) { type_ids.push_back(col.type().id()); }

    table_schemas.push_back({
      {{table, names}},          // std::vector<ral::frame::BlazingTableView> blazingTableViews
      type_ids,                  // std::vector<cudf::type_id> types
      {},                        // std::vector<std::string> files
      {},                        // std::vector<std::string> datasource
      names,                     // std::vector<std::string> names
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

  std::cout << "got here!" << std::endl;

  for (std::size_t i = 0; i < schemas.Length(); ++i) {
    NapiToCPP::Object schema = schemas.Get(i);

    std::vector<std::string> names               = schema.Get("names");
    std::vector<std::string> files               = schema.Get("files");
    std::vector<size_t> calcite_to_file_indicies = schema.Get("calciteToFileIndicies");

    std::vector<int32_t> type_ints = schema.Get("types");
    std::vector<cudf::type_id> type_ids;
    type_ids.reserve(type_ints.size());
    for (auto const& type : type_ints) { type_ids.push_back(cudf::type_id(type)); }

    bool has_header_csv = schema.Get("hasHeaderCSV");

    table_schemas.push_back({
      {},                        // std::vector<ral::frame::BlazingTableView> blazingTableViews
      type_ids,                  // std::vector<cudf::type_id> types
      files,                     // std::vector<std::string> files
      files,                     // std::vector<std::string> datasource
      names,                     // std::vector<std::string> names
      calcite_to_file_indicies,  // std::vector<size_t> calcite_to_file_indices
      {},                        // std::vector<bool> in_file
      ral::io::DataType::CSV,    // int data_type
      has_header_csv,            // bool has_header_csv = false
      {cudf::table_view{}, {}},  // ral::frame::BlazingTableView metadata
      {{0}},                     // std::vector<std::vector<int>> row_groups_ids
      nullptr                    // std::shared_ptr<arrow::Table> arrow_tabl
    });
    table_schema_cpp_arg_keys.push_back({"has_header_csv"});
    table_schema_cpp_arg_values.push_back({has_header_csv ? "True" : "False"});
    files_all.push_back(files);
    file_types.push_back(ral::io::DataType::CSV);
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

  return ExecutionGraph::New(env, result, context);
}

std::string run_generate_physical_graph(uint32_t const& masterIndex,
                                        std::vector<std::string> const& worker_ids,
                                        int32_t const& ctx_token,
                                        std::string const& query) {
  return ::runGeneratePhysicalGraph(masterIndex, worker_ids, ctx_token, query);
}

Napi::Value parse_schema(Napi::Env const& env,
                         std::vector<std::string> const& input,
                         std::string const& file_format,
                         bool const& ignoreMissingFiles) {
  auto table_schema = ::parseSchema(input, file_format, {}, {}, {}, ignoreMissingFiles);

  auto result = Napi::Object::New(env);

  auto files = Napi::Array::New(env, table_schema.files.size());
  for (size_t i = 0; i < table_schema.files.size(); ++i) {
    files.Set(i, Napi::String::New(env, table_schema.files[i]));
  }
  result.Set("files", files);

  result.Set("fileType", table_schema.data_type);

  auto types = Napi::Array::New(env, table_schema.types.size());
  for (size_t i = 0; i < table_schema.types.size(); ++i) {
    types.Set(i, cudf_to_arrow_type(env, cudf::data_type(table_schema.types[i])));
  }
  result.Set("types", types);

  auto names = Napi::Array::New(env, table_schema.names.size());
  for (size_t i = 0; i < table_schema.names.size(); ++i) {
    names.Set(i, Napi::String::New(env, table_schema.names[i]));
  }
  result.Set("names", names);

  auto calcite_to_file_indices = Napi::Array::New(env, table_schema.calcite_to_file_indices.size());
  for (size_t i = 0; i < table_schema.calcite_to_file_indices.size(); ++i) {
    calcite_to_file_indices.Set(i, Napi::Number::New(env, table_schema.calcite_to_file_indices[i]));
  }
  result.Set("calciteToFileIndicies", calcite_to_file_indices);

  result.Set("hasHeaderCSV", Napi::Boolean::New(env, table_schema.has_header_csv));

  return result;
}

void start_execute_graph(std::shared_ptr<ral::cache::graph> const& graph) {
  ::startExecuteGraph(graph, graph->get_context_token());
}

std::tuple<std::vector<std::string>, std::vector<std::unique_ptr<cudf::table>>>
get_execute_graph_result(std::shared_ptr<ral::cache::graph> const& graph) {
  auto bsql_result = std::move(::getExecuteGraphResult(graph, graph->get_context_token()));
  return {std::move(bsql_result->names), std::move(bsql_result->cudfTables)};
}

}  // namespace blazingsql
}  // namespace nv
