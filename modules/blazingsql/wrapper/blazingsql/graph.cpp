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

#include <nv_node/utilities/args.hpp>

#include <execution_graph/graph.h>

namespace nv {

Napi::Function ExecutionGraph::Init(Napi::Env env, Napi::Object exports) {
  return DefineClass(env, "ExecutionGraph", {});
}

ExecutionGraph::wrapper_t ExecutionGraph::New(Napi::Env const& env,
                                              std::shared_ptr<ral::cache::graph> graph) {
  auto inst    = EnvLocalObjectWrap<ExecutionGraph>::New(env, {});
  inst->_graph = graph;
  return inst;
}

ExecutionGraph::ExecutionGraph(Napi::CallbackInfo const& info)
  : EnvLocalObjectWrap<ExecutionGraph>(info) {}

}  // namespace nv
