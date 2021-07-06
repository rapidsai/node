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

CacheMachine::CacheMachine(Napi::CallbackInfo const& info)
  : EnvLocalObjectWrap<CacheMachine>(info) {}

}  // namespace nv
