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

#include <napi.h>
#include <nv_node/objectwrap.hpp>

typedef struct ucp_context* ucp_context_h;

namespace nv {

struct UcpContext : public EnvLocalObjectWrap<UcpContext> {
  /**
   * @brief Initialize and export the UcpContext JavaScript constructor and prototype.
   *
   * @param env The active JavaScript environment.
   * @param exports The exports object to decorate.
   * @return Napi::Function The UcpContext constructor function.
   */
  static Napi::Function Init(Napi::Env env, Napi::Object exports);
  /**
   * @brief Construct a new UcpContext instance from a ral::cache::graph.
   *
   * @param cache The shared pointer to the UcpContext.
   */
  static wrapper_t New(Napi::Env const& env);

  /**
   * @brief Construct a new UcpContext instance from JavaScript.
   */
  UcpContext(Napi::CallbackInfo const& info);

  inline operator ucp_context_h() { return _context; }

 private:
  ucp_context_h _context;
};

}  // namespace nv
