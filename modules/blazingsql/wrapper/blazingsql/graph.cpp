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

#include "graph.hpp"

#include <node_cudf/table.hpp>
#include <nv_node/utilities/args.hpp>
#include "api.hpp"

#include <execution_graph/graph.h>

namespace nv {

Napi::Function ExecutionGraph::Init(Napi::Env env, Napi::Object exports) {
  return DefineClass(env, "ExecutionGraph", {InstanceMethod<&ExecutionGraph::start>("start"),InstanceMethod<&ExecutionGraph::result>("result")});
}

ExecutionGraph::wrapper_t ExecutionGraph::New(Napi::Env const& env,
                                              std::shared_ptr<ral::cache::graph> graph) {
  auto inst    = EnvLocalObjectWrap<ExecutionGraph>::New(env, {});
  inst->_graph = graph;
  return inst;
}

ExecutionGraph::ExecutionGraph(Napi::CallbackInfo const& info)
  : EnvLocalObjectWrap<ExecutionGraph>(info) {}

void ExecutionGraph::start(Napi::CallbackInfo const& info) {
  if (!_started) { start_execute_graph(*this, _graph->get_context_token()); }
}

Napi::Value ExecutionGraph::result(Napi::CallbackInfo const& info) {
  Napi::Env env = info.Env();
  start(info);
  if (!_results) {
    auto [names, tables] = nv::get_execute_graph_result(*this, _graph->get_context_token());
    _names               = std::move(names);
    _tables              = std::move(tables);
  }

    auto result_names = Napi::Array::New(env, _names.size());
    for (size_t i = 0; i < _names.size(); ++i) {
      result_names.Set(i, Napi::String::New(env, _names[i]));
    }

    auto result_tables = Napi::Array::New(env, _tables.size());
    for (size_t i = 0; i < _tables.size(); ++i) {
      result_tables.Set(i, nv::Table::New(env, std::move(_tables[i])));
    }

    auto result = Napi::Object::New(env);
    result.Set("names", result_names);
    result.Set("tables", result_tables);
    return result;
}

Napi::Value ExecutionGraph::send_to(Napi::CallbackInfo const& info) {
  auto df                = result(info);
  int target_ral_id      = info[0].ToNumber();  // TODO uint16_t
  std::string message_id = info[1].ToString();

  auto query_context = _graph->get_last_kernel()->output_cache()->get_context();

  // todo: add to cache
  return this->Value();
}

}  // namespace nv
