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

#include <nv_node/objectwrap.hpp>

#include <napi.h>

namespace nv {

struct CacheMachine;

struct Context : public EnvLocalObjectWrap<Context> {
  /**
   * @brief Initialize and export the Context JavaScript constructor and prototype.
   *
   * @param env The active JavaScript environment.
   * @param exports The exports object to decorate.
   * @return Napi::Function The Context constructor function.
   */
  static Napi::Function Init(Napi::Env env, Napi::Object exports);

  /**
   * @brief Construct a new Context instance from existing device memory.
   *
   * @return wrapper_t The new Context instance
   */
  static wrapper_t New(Napi::Env const& env);

  /**
   * @brief Construct a new Context instance from JavaScript.
   */
  Context(Napi::CallbackInfo const& info);

  /**
   * @brief Returns the port the context is operating on
   */
  int32_t port() const noexcept { return _port; }

 private:
  int32_t _port{};
  Napi::Reference<Wrapper<CacheMachine>> _transport_out;
  Napi::Reference<Wrapper<CacheMachine>> _transport_in;

  Napi::Value port(Napi::CallbackInfo const& info);
};

}  // namespace nv
