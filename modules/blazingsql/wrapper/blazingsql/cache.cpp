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

#include "cache.hpp"

#include <nv_node/utilities/args.hpp>

#include <cache_machine/CacheMachine.h>

namespace nv {

Napi::Function CacheMachine::Init(Napi::Env env, Napi::Object exports) {
  return DefineClass(env, "CacheMachine", {});
}

CacheMachine::wrapper_t CacheMachine::New(Napi::Env const& env,
                                          std::shared_ptr<ral::cache::CacheMachine> cache) {
  auto inst    = EnvLocalObjectWrap<CacheMachine>::New(env, {});
  inst->_cache = cache;
  return inst;
}

void CacheMachine::add_to_cache(blazingdb::manager::Context* context,
                                std::string const& message_id,
                                uint16_t const& ral_id,
                                std::vector<std::string> const& column_names,
                                cudf::table_view const& table_view) {
  std::unique_ptr<ral::frame::BlazingTable> table =
    std::make_unique<ral::frame::BlazingTable>(table_view, column_names);

  ral::cache::MetadataDictionary metadata;

  metadata.add_value(
    ral::cache::RAL_ID_METADATA_LABEL,
    context->getNodeIndex(ral::communication::CommunicationData::getInstance().getSelfNode()));
  metadata.add_value(ral::cache::KERNEL_ID_METADATA_LABEL, std::to_string(0));  // unused
  metadata.add_value(ral::cache::QUERY_ID_METADATA_LABEL,
                     std::to_string(context->getContextToken()));
  metadata.add_value(ral::cache::ADD_TO_SPECIFIC_CACHE_METADATA_LABEL, "false");
  metadata.add_value(ral::cache::CACHE_ID_METADATA_LABEL, 0);  // unused, potentially unset
  metadata.add_value(ral::cache::SENDER_WORKER_ID_METADATA_LABEL, ral_id);
  metadata.add_value(ral::cache::WORKER_IDS_METADATA_LABEL, ral_id);
  metadata.add_value(ral::cache::UNIQUE_MESSAGE_ID, message_id);

  this->_cache->addToCache(std::move(table), message_id, true, metadata, true);
}

std::tuple<std::vector<std::string>, std::unique_ptr<cudf::table>> CacheMachine::pull_from_cache(
  std::string const& message_id) {
  auto result = this->_cache->pullCacheData(message_id);
  return {std::move(result->names()), std::move(result->decache()->releaseCudfTable())};
}

CacheMachine::CacheMachine(Napi::CallbackInfo const& info)
  : EnvLocalObjectWrap<CacheMachine>(info) {}

}  // namespace nv
