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

#include "contextwrapper.hpp"
#include "graph.hpp"

#include <nv_node/utilities/args.hpp>

struct NodeMetaDataUCP;
struct TableSchema;

namespace nv {

ContextWrapper::wrapper_t initialize(Napi::Env const& env, NapiToCPP::Object const& props);

Napi::Value get_table_scan_info(Napi::CallbackInfo const& info);

ExecutionGraph::wrapper_t run_generate_graph(Napi::CallbackInfo const& info);

void start_execute_graph(Napi::CallbackInfo const& info);

Napi::Value get_execute_graph_result(Napi::CallbackInfo const& info);

}  // namespace nv
