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

#include "async.hpp"
#include "cache.hpp"
#include "ucpcontext.hpp"

#include <nv_node/objectwrap.hpp>

namespace nv {
namespace blazingsql {

struct CacheMachine;

struct Context : public EnvLocalObjectWrap<Context> {
  /**
   * @brief Initialize and export the ContextWrapper JavaScript constructor and prototype.
   *
   * @param env The active JavaScript environment.
   * @param exports The exports object to decorate.
   * @return Napi::Function The ContextWrapper constructor function.
   */
  static Napi::Function Init(Napi::Env const& env, Napi::Object exports);

  /**
   * @brief Construct a new ContextWrapper instance from existing device memory.
   *
   * @return wrapper_t The new ContextWrapper instance
   */
  static wrapper_t New(Napi::Env const& env,
                       int32_t const& ral_id,
                       std::pair<std::pair<std::shared_ptr<ral::cache::CacheMachine>,
                                           std::shared_ptr<ral::cache::CacheMachine>>,
                                 int> const& pair,
                       UcpContext::wrapper_t const& ucp_context);

  /**
   * @brief Construct a new ContextWrapper instance from JavaScript.
   */
  Context(Napi::CallbackInfo const& info);

  inline int32_t get_ral_id() const { return _id; }

  inline int32_t get_node_id() const { return _node_id; }
  inline void set_node_id(int32_t node_id) { _node_id = node_id; }

  void send(int32_t const& dst_ral_id,
            std::string const& ctx_token,
            std::string const& message_id,
            std::vector<std::string> const& column_names,
            cudf::table_view const& table_view);

  SQLTask* pull(std::vector<std::string> const& message_ids);

 private:
  int32_t _id{};
  int32_t _port{};
  int32_t _node_id{-1};
  std::vector<int32_t> _worker_ids{};
  Napi::Reference<UcpContext::wrapper_t> _ucp_context;
  Napi::Reference<Wrapper<CacheMachine>> _transport_in;
  Napi::Reference<Wrapper<CacheMachine>> _transport_out;

  void send(Napi::CallbackInfo const& info);
  Napi::Value pull(Napi::CallbackInfo const& info);
  Napi::Value broadcast(Napi::CallbackInfo const& info);
  Napi::Value run_generate_graph(Napi::CallbackInfo const& info);
};

}  // namespace blazingsql
}  // namespace nv
