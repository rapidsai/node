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

#include <typeinfo>

#include <napi.h>

namespace nv {

struct EnvLocalAddon {
  inline EnvLocalAddon(Napi::Env const& env, Napi::Object exports) {
    _cpp_exports = Napi::Persistent(Napi::Object::New(env));
  };

  inline Napi::Value GetCppExports(Napi::CallbackInfo const& info) { return _cpp_exports.Value(); }

  template <typename Class>
  inline Napi::Function GetConstructor() {
    return _cpp_exports.Get(typeid(Class).hash_code()).As<Napi::Function>();
  }

 protected:
  Napi::ObjectReference _cpp_exports;
  Napi::FunctionReference _after_init;

  template <typename Class>
  inline Napi::Function InitClass(Napi::Env const& env, Napi::Object exports) {
    auto const constructor = Class::Init(env, exports);
    _cpp_exports.Set(typeid(Class).hash_code(), constructor);
    return constructor;
  }

  inline Napi::Value InitAddon(Napi::CallbackInfo const& info) {
    std::vector<napi_value> args;
    args.reserve(info.Length());
    for (std::size_t i = 0; i < info.Length(); ++i) { args.push_back(info[i]); }
    RegisterAddon(info.Env(), args);
    if (!_after_init.IsEmpty()) {  //
      _after_init.Call(info.This(), args);
    }
    return info.This();
  }

  inline void RegisterAddon(Napi::Env env, std::vector<napi_value> const& info) {
    auto rhs          = env.GetInstanceData<EnvLocalAddon>()->_cpp_exports.Value();
    auto GlobalObject = env.Global().Get("Object").As<Napi::Function>();
    auto ObjectAssign = GlobalObject.Get("assign").As<Napi::Function>();

    for (auto const& _ : info) {
      Napi::HandleScope scope(env);
      Napi::Value const module(env, _);
      if (module.IsObject()) {
        auto const addon = module.As<Napi::Object>();
        if (addon.Has("_cpp_exports")) {
          auto const exports = addon.Get("_cpp_exports");
          if (exports.Type() == napi_object) {  //
            ObjectAssign({rhs, exports.As<Napi::Object>()});
          }
        }
      }
    }
  }
};

}  // namespace nv
