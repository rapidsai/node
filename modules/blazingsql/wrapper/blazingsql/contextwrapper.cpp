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

namespace nv {

Napi::Function ContextWrapper::Init(Napi::Env env, Napi::Object exports) {
  return DefineClass(env, "ContextWrapper", {});
}

ContextWrapper::wrapper_t ContextWrapper::New(
  Napi::Env const& env,
  std::pair<
    std::pair<std::shared_ptr<ral::cache::CacheMachine>, std::shared_ptr<ral::cache::CacheMachine>>,
    int> args) {
  auto inst    = EnvLocalObjectWrap<ContextWrapper>::New(env, {});
  auto& caches = args.first;
  inst->_port  = args.second;
  return inst;
}

ContextWrapper::ContextWrapper(Napi::CallbackInfo const& info)
  : EnvLocalObjectWrap<ContextWrapper>(info) {}

}  // namespace nv
