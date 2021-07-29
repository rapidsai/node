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

#include "contextwrapper.hpp"
#include <node_cudf/table.hpp>

namespace nv {

Napi::Function ContextWrapper::Init(Napi::Env env, Napi::Object exports) {
  return DefineClass(env, "ContextWrapper", {});
}

ContextWrapper::wrapper_t ContextWrapper::New(
  Napi::Env const& env,
  std::pair<
    std::pair<std::shared_ptr<ral::cache::CacheMachine>, std::shared_ptr<ral::cache::CacheMachine>>,
    int> pair,
  Napi::Object const& ucp_context) {
  auto inst            = EnvLocalObjectWrap<ContextWrapper>::New(env, {});
  auto& caches         = pair.first;
  inst->_port          = pair.second;
  inst->_transport_in  = Napi::Persistent(CacheMachine::New(env, caches.first));
  inst->_transport_out = Napi::Persistent(CacheMachine::New(env, caches.second));
  inst->_ucp_context   = Napi::Persistent(ucp_context);
  return inst;
}

ContextWrapper::ContextWrapper(Napi::CallbackInfo const& info)
  : EnvLocalObjectWrap<ContextWrapper>(info) {}

void ContextWrapper::add_to_cache(std::string const& message_id,
                                  std::vector<std::string> const& column_names,
                                  cudf::table_view const& table_view) {
  this->_transport_out.Value()->add_to_cache(message_id, column_names, table_view);
}

std::tuple<std::vector<std::string>, std::unique_ptr<cudf::table>> ContextWrapper::pull_from_cache(
  std::string const& message_id) {
  return this->_transport_in.Value()->pull_from_cache(message_id);
}

}  // namespace nv
