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

#include "context.hpp"
#include "graph.hpp"

#include <cudf/table/table.hpp>

#include <nv_node/utilities/args.hpp>

struct NodeMetaDataUCP;
struct TableSchema;

namespace nv {
namespace blazingsql {

std::tuple<uint16_t,
           int32_t,
           std::vector<int32_t>,
           UcpContext::wrapper_t,
           std::shared_ptr<ral::cache::CacheMachine>,
           std::shared_ptr<ral::cache::CacheMachine>>
initialize(Napi::Env const& env, NapiToCPP::Object const& props);

std::tuple<std::vector<std::string>, std::vector<std::string>> get_table_scan_info(
  std::string const& logical_plan);

ExecutionGraph::wrapper_t run_generate_graph(
  Napi::Env const& env,
  Wrapper<Context> const& context,
  uint32_t const& masterIndex,
  std::vector<std::string> const& worker_ids,
  std::vector<cudf::table_view> const& table_views,
  std::vector<std::vector<std::string>> const& column_names,
  std::vector<std::string> const& table_names,
  std::vector<std::string> const& table_scans,
  int32_t const& ctx_token,
  std::string const& query,
  std::string const& sql,
  std::string const& current_timestamp,
  std::map<std::string, std::string> const& config_options);

std::string run_generate_physical_graph(uint32_t const& masterIndex,
                                        std::vector<std::string> const& worker_ids,
                                        int32_t const& ctx_token,
                                        std::string const& query);

Napi::Value parse_schema(Napi::Env const& env,
                         std::vector<std::string> const& input,
                         std::string const& file_format,
                         bool const& ignoreMissingFiles);

void start_execute_graph(std::shared_ptr<ral::cache::graph> const& execution_graph);

std::tuple<std::vector<std::string>, std::vector<std::unique_ptr<cudf::table>>>
get_execute_graph_result(std::shared_ptr<ral::cache::graph> const& execution_graph);

}  // namespace blazingsql
}  // namespace nv
