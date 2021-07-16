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

#pragma once

#include "contextwrapper.hpp"
#include "graph.hpp"

#include <nv_node/utilities/args.hpp>

#include <cudf/table/table.hpp>

struct NodeMetaDataUCP;
struct TableSchema;

namespace nv {

ContextWrapper::wrapper_t initialize(Napi::Env const& env, NapiToCPP::Object const& props);

std::tuple<std::vector<std::string>, std::vector<std::string>> get_table_scan_info(
  std::string const& logical_plan);

ExecutionGraph::wrapper_t run_generate_graph(Napi::Env env,
                                             uint32_t masterIndex,
                                             std::vector<std::string> worker_ids,
                                             std::vector<cudf::table_view> table_views,
                                             std::vector<std::vector<std::string>> column_names,
                                             std::vector<std::string> table_names,
                                             std::vector<std::string> table_scans,
                                             int32_t ctx_token,
                                             std::string query,
                                             std::string sql,
                                             std::string current_timestamp,
                                             std::map<std::string, std::string> config_options);

std::string run_generate_physical_graph(uint32_t masterIndex,
                                        std::vector<std::string> worker_ids,
                                        int32_t ctx_token,
                                        std::string query);

void start_execute_graph(ExecutionGraph::wrapper_t const& execution_graph, int32_t const ctx_token);

std::tuple<std::vector<std::string>, std::vector<std::unique_ptr<cudf::table>>>
get_execute_graph_result(ExecutionGraph::wrapper_t const& execution_graph, int32_t const ctx_token);

}  // namespace nv
