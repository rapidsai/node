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
#include <node_cudf/table.hpp>

namespace nv {

Napi::Function Context::Init(Napi::Env env, Napi::Object exports) {
  return DefineClass(env,
                     "Context",
                     {InstanceMethod<&Context::add_to_cache>("addToCache"),
                      InstanceMethod<&Context::pull_from_cache>("pullFromCache")});
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

void Context::add_to_cache(Napi::CallbackInfo const& info) {
  nv::CallbackArgs args{info};

  std::string message_id = args[0];

  nv::NapiToCPP::Object df       = args[1];
  std::vector<std::string> names = df.Get("names");
  Napi::Function asTable         = df.Get("asTable");
  nv::Table::wrapper_t table     = asTable.Call(df.val, {}).ToObject();

  this->context.Value()->add_to_cache(message_id, names, table->view());
}

Napi::Value Context::pull_from_cache(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  nv::CallbackArgs args{info};

  std::string message_id = args[0];

  auto [bsql_names, bsql_table] = this->context.Value()->pull_from_cache(message_id);

  auto result_names = Napi::Array::New(env, bsql_names.size());
  for (size_t i = 0; i < bsql_names.size(); ++i) {
    result_names.Set(i, Napi::String::New(env, bsql_names[i]));
  }

  auto result = Napi::Object::New(env);
  result.Set("names", result_names);
  result.Set("table", nv::Table::New(env, std::move(bsql_table)));
  return result;
}

}  // namespace nv
