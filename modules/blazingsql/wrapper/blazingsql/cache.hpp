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

#include <node_cudf/table.hpp>

#include <nv_node/objectwrap.hpp>

#include <napi.h>

namespace ral {
namespace cache {
struct CacheMachine;
}
}  // namespace ral

namespace nv {

struct CacheMachine : public nv::EnvLocalObjectWrap<CacheMachine> {
  /**
   * @brief Initialize and export the CacheMachine JavaScript constructor and prototype.
   *
   * @param env The active JavaScript environment.
   * @param exports The exports object to decorate.
   * @return Napi::Function The CacheMachine constructor function.
   */
  static Napi::Function Init(Napi::Env const& env, Napi::Object exports);
  /**
   * @brief Construct a new CacheMachine instance from a ral::cache::CacheMachine.
   *
   * @param cache The shared pointer to the CacheMachine.
   */
  static wrapper_t New(Napi::Env const& env,
                       std::shared_ptr<ral::cache::CacheMachine> const& cache);
  /**
   * @brief Construct a new CacheMachine instance from JavaScript.
   */
  CacheMachine(Napi::CallbackInfo const& info);

  inline operator std::shared_ptr<ral::cache::CacheMachine>() { return _cache; }

  void add_to_cache(int32_t const& node_id,
                    int32_t const& src_ral_id,
                    int32_t const& dst_ral_id,
                    std::string const& ctx_token,
                    std::string const& message_id,
                    std::vector<std::string> const& column_names,
                    cudf::table_view const& table_view);

  std::tuple<std::vector<std::string>, std::unique_ptr<cudf::table>> pull_from_cache(
    std::string const& message_id);

 private:
  std::shared_ptr<ral::cache::CacheMachine> _cache;
};

}  // namespace nv
