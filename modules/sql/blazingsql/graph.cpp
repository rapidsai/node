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

#include "blazingsql_wrapper/graph.hpp"
#include "blazingsql_wrapper/api.hpp"
#include "blazingsql_wrapper/async.hpp"

#include <node_cudf/table.hpp>

#include <cudf/concatenate.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <nv_node/utilities/args.hpp>

#include <execution_graph/graph.h>

namespace nv {
namespace blazingsql {

Napi::Function ExecutionGraph::Init(Napi::Env const& env, Napi::Object exports) {
  return DefineClass(env,
                     "ExecutionGraph",
                     {
                       InstanceMethod<&ExecutionGraph::send>("sendTo"),
                       InstanceMethod<&ExecutionGraph::start>("start"),
                       InstanceMethod<&ExecutionGraph::result>("result"),
                     });
}

ExecutionGraph::wrapper_t ExecutionGraph::New(Napi::Env const& env,
                                              std::shared_ptr<ral::cache::graph> const& graph,
                                              nv::Wrapper<Context> const& context) {
  auto inst      = EnvLocalObjectWrap<ExecutionGraph>::New(env, {});
  inst->_graph   = graph;
  inst->_context = Napi::Persistent(context);
  if (context->get_node_id() == -1) {
    context->set_node_id(graph->get_last_kernel()->input_cache()->get_context()->getNodeIndex(
      ral::communication::CommunicationData::getInstance().getSelfNode()));
  }
  return inst;
}

ExecutionGraph::ExecutionGraph(Napi::CallbackInfo const& info)
  : EnvLocalObjectWrap<ExecutionGraph>(info) {}

void ExecutionGraph::start(Napi::CallbackInfo const& info) {
  if (!_started) {
    start_execute_graph(_graph);
    _started = true;
  }
}

Napi::Value ExecutionGraph::result(Napi::CallbackInfo const& info) {
  auto env = info.Env();

  start(info);

  if (_fetched == false) {
    _fetched  = true;
    auto task = new SQLTask(env, [this]() {
      auto [names, tables] = std::move(get_execute_graph_result(_graph));
      return std::make_pair(std::move(names), std::move(tables));
    });
    _results  = Napi::Persistent(task->run());
  }

  return _results.Value();
}

Napi::Value ExecutionGraph::send(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};

  int32_t dst_ral_id      = args[0];
  Napi::Array data_frames = args[1];
  int32_t nonce           = args[2];

  auto messages = Napi::Array::New(env, data_frames.Length());
  for (int i = 0; i < data_frames.Length(); ++i) {
    NapiToCPP::Object df           = data_frames.Get(i);
    std::vector<std::string> names = df.Get("names");
    Napi::Function asTable         = df.Get("asTable");

    Table::wrapper_t table = asTable.Call(df.val, {}).ToObject();

    auto ctx_token =
      std::to_string(_graph->get_last_kernel()->input_cache()->get_context()->getContextToken());

    std::string message =
      "broadcast_table_message_" + std::to_string(nonce) + "_" + std::to_string(i);
    messages[i] = message;

    _context.Value()->send(dst_ral_id, ctx_token, message, names, *table);
  }

  return messages;
}

}  // namespace blazingsql
}  // namespace nv
