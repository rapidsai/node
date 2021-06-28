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

// #pragma once

// #include "cachemachine.hpp"

// #include <engine/initialize.h>

// namespace nv {
// namespace blazingsql {

// // in engine/initialize.h
// std::tuple<CacheMachine::wrapper_t, CacheMachine::wrapper_t, int32_t> initialize(
//   uint16_t ralId,
//   std::string worker_id,
//   std::string network_iface_name,
//   int ralCommunicationPort,
//   std::vector<NodeMetaDataUCP> workers_ucp_info,
//   bool singleNode,
//   std::map<std::string, std::string> config_options,
//   std::string allocation_mode,
//   std::size_t initial_pool_size,
//   std::size_t maximum_pool_size,
//   bool enable_logging);

// // in engine/engine.h
// void runGenerateGraph();
// void startExecuteGraph();
// void getExecuteGraphResult();

// }  // namespace blazingsql
// }  // namespace nv
