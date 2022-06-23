// Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <nv_node/addon.hpp>
#include <nv_node/utilities/args.hpp>
#include <nv_node/utilities/napi_to_cpp.hpp>

#include <napi.h>

std::ostream& operator<<(std::ostream& os, nv::NapiToCPP const& self) {
  return os << self.operator std::string();
};

struct rapidsai_core : public nv::EnvLocalAddon, public Napi::Addon<rapidsai_core> {
  rapidsai_core(Napi::Env const& env, Napi::Object exports) : EnvLocalAddon(env, exports) {
    DefineAddon(exports,
                {
                  InstanceValue("_cpp_exports", _cpp_exports.Value()),
                  InstanceMethod("init", &rapidsai_core::InitAddon),
                });
  }
};

NODE_API_ADDON(rapidsai_core);
