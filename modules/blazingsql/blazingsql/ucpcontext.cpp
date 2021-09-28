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

#include "blazingsql_wrapper/ucpcontext.hpp"

#include <nv_node/utilities/args.hpp>

#include <communication/ucx_init.h>

namespace nv {
namespace blazingsql {

Napi::Function UcpContext::Init(Napi::Env const& env, Napi::Object exports) {
  return DefineClass(env, "UcpContext", {});
}

UcpContext::wrapper_t UcpContext::New(Napi::Env const& env) {
  return EnvLocalObjectWrap<UcpContext>::New(env, {});
}

UcpContext::UcpContext(Napi::CallbackInfo const& info) : EnvLocalObjectWrap<UcpContext>(info) {
  this->_ucp_context = ral::communication::CreateUcpContext();
}

}  // namespace blazingsql
}  // namespace nv
