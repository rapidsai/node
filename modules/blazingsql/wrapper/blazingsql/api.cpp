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

#pragma once

#include "api.hpp"

#include <engine/common.h>
#include <engine/engine.h>
#include <engine/initialize.h>
#include <io/io.h>

namespace nv {

Context::wrapper_t initialize(
  Napi::Env const& env,
  uint16_t ral_id,
  std::string worker_id,
  std::string network_iface_name,
  int ral_communication_port,
  std::vector<NodeMetaDataUCP>
    workers_ucp_info,  // this Array has Objects that describe NodeMetaDataUCP fields
  bool single_node,
  std::map<std::string, std::string> config_options,
  std::string allocation_mode,
  std::size_t initial_pool_size,
  std::size_t maximum_pool_size,
  bool enable_logging) {
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

  auto& caches = init_result.first;

  auto opts = Napi::Object::New(env);
  // TODO: Set the results of caches, etc here before returning.
  return EnvLocalObjectWrap<Context>::New(env, {opts});
}

ExecutionGraph::wrapper_t run_generate_graph(
  Napi::Env const& env,
  uint32_t master_index,
  std::vector<std::string> worker_ids,
  std::vector<std::string> table_names,
  std::vector<std::string> table_scans,
  std::vector<TableSchema> table_schemas,
  std::vector<std::vector<std::string>> table_schema_keys,
  std::vector<std::vector<std::string>> table_schema_values,
  std::vector<std::vector<std::string>> files_all,
  std::vector<int> file_types,
  int32_t ctx_token,
  std::string query,
  std::vector<std::vector<std::map<std::string, std::string>>> uri_values,
  std::map<std::string, std::string> config_options,
  std::string sql,
  std::string current_timestamp) {
  auto result = ::runGenerateGraph(master_index,
                                   worker_ids,
                                   table_names,
                                   table_scans,
                                   table_schemas,
                                   table_schema_keys,
                                   table_schema_values,
                                   files_all,
                                   file_types,
                                   ctx_token,
                                   query,
                                   uri_values,
                                   config_options,
                                   sql,
                                   current_timestamp);
  auto opts   = Napi::Object::New(env);
  // TODO: Set the results of caches, etc here before returning.
  return EnvLocalObjectWrap<ExecutionGraph>::New(env, {opts});
}

void start_execute_graph(ExecutionGraph::wrapper_t graph, int32_t ctx_token) {
  // TODO: Figure out how to get _graph from graph.
  // ::startExecuteGraph(, ctx_token);
}

ExecutionGraph::wrapper_t get_execute_graph_result(Napi::Env const& env,
                                                   ExecutionGraph::wrapper_t graph,
                                                   int32_t ctx_token) {
  // TODO: Figure out how to get _graph from graph.
  // auto result = ::getExecuteGraphResult(graph, ctx_token);

  auto opts = Napi::Object::New(env);
  return EnvLocalObjectWrap<ExecutionGraph>::New(env, {opts});
}

}  // namespace nv
