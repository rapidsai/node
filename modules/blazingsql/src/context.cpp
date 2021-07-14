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

#include "context.hpp"

#include <blazingsql/api.hpp>
#include <blazingsql/cache.hpp>
#include <blazingsql/graph.hpp>

#include <nv_node/utilities/args.hpp>

namespace nv {

Napi::Function Context::Init(Napi::Env env, Napi::Object exports) {
  return DefineClass(env, "Context", {});
}

Context::wrapper_t Context::New(Napi::Env const& env) {
  return EnvLocalObjectWrap<Context>::New(env);
}

Context::Context(Napi::CallbackInfo const& info) : EnvLocalObjectWrap<Context>(info) {
  auto env                = info.Env();
  NapiToCPP::Object props = info[0];
  auto result_context     = nv::initialize(env, props);
  this->context           = Napi::Persistent(result_context);
}

}  // namespace nv
