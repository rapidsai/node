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

#include "blazingsql_wrapper/context.hpp"
#include "blazingsql_wrapper/api.hpp"
#include "blazingsql_wrapper/cache.hpp"

#include <node_cudf/table.hpp>

#include <execution_graph/Context.h>

namespace nv {
namespace blazingsql {

Napi::Function Context::Init(Napi::Env const& env, Napi::Object exports) {
  return DefineClass(env,
                     "Context",
                     {
                       InstanceMethod<&Context::send>("send"),
                       InstanceMethod<&Context::pull>("pull"),
                       InstanceMethod<&Context::run_generate_graph>("runGenerateGraph"),
                     });
}

Context::Context(Napi::CallbackInfo const& info) : EnvLocalObjectWrap<Context>(info) {
  auto env             = info.Env();
  auto result          = blazingsql::initialize(env, info[0]);
  this->_id            = std::get<0>(result);
  this->_port          = std::get<1>(result);
  this->_worker_ids    = std::move(std::get<2>(result));
  this->_ucp_context   = Napi::Persistent(std::get<3>(result));
  this->_transport_in  = Napi::Persistent(CacheMachine::New(env, std::get<4>(result)));
  this->_transport_out = Napi::Persistent(CacheMachine::New(env, std::get<5>(result)));
}

Napi::Value Context::run_generate_graph(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};

  Napi::Array data_frames              = args[0];
  std::vector<std::string> table_names = args[1];
  std::vector<std::string> table_scans = args[2];
  int32_t ctx_token                    = args[3];
  std::string query                    = args[4];
  std::string sql                      = args[5];
  std::string current_timestamp        = args[6];
  auto config_options                  = [&] {
    std::map<std::string, std::string> config{};
    auto prop = args[7];
    if (!prop.IsNull() && prop.IsObject()) {
      auto opts = prop.As<Napi::Object>();
      auto keys = opts.GetPropertyNames();
      for (auto i = 0u; i < keys.Length(); ++i) {
        std::string name = keys.Get(i).ToString();
        config[name]     = opts.Get(name).ToString();
        if (config[name] == "true") {
          config[name] = "True";
        } else if (config[name] == "false") {
          config[name] = "False";
        }
      }
    }
    return config;
  }();

  std::vector<cudf::table_view> table_views;
  std::vector<std::vector<std::string>> column_names;

  table_views.reserve(data_frames.Length());
  column_names.reserve(data_frames.Length());

  auto tables = Napi::Array::New(env, data_frames.Length());

  for (std::size_t i = 0; i < data_frames.Length(); ++i) {
    NapiToCPP::Object df           = data_frames.Get(i);
    std::vector<std::string> names = df.Get("names");
    Napi::Function asTable         = df.Get("asTable");
    Table::wrapper_t table         = asTable.Call(df.val, {}).ToObject();

    tables.Set(i, table);
    table_views.push_back(*table);
    column_names.push_back(names);
  }

  return blazingsql::run_generate_graph(env,
                                        *this,
                                        0,
                                        this->_worker_ids,
                                        table_views,
                                        column_names,
                                        table_names,
                                        table_scans,
                                        ctx_token,
                                        query,
                                        sql,
                                        current_timestamp,
                                        config_options);
}

void Context::send(int32_t const& dst_ral_id,
                   std::string const& ctx_token,
                   std::string const& message_id,
                   std::vector<std::string> const& column_names,
                   cudf::table_view const& table_view) {
  this->_transport_out.Value()->add_to_cache(
    get_node_id(), get_ral_id(), dst_ral_id, ctx_token, message_id, column_names, table_view);
}

std::tuple<std::vector<std::string>, std::unique_ptr<cudf::table>> Context::pull(
  std::string const& message_id) {
  return std::move(this->_transport_in.Value()->pull_from_cache(message_id));
}

void Context::send(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};

  int32_t dst_ral_id     = args[0];
  std::string ctx_token  = args[1];
  std::string message_id = args[2];
  NapiToCPP::Object df   = args[3];

  std::vector<std::string> names = df.Get("names");
  Napi::Function asTable         = df.Get("asTable");
  Table::wrapper_t table         = asTable.Call(df.val, {}).ToObject();

  this->send(dst_ral_id, ctx_token, message_id, names, *table);
}

Napi::Value Context::pull(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};

  std::string message_id = args[0];

  auto [bsql_names, bsql_table] = pull(message_id);

  auto result_names = Napi::Array::New(env, bsql_names.size());
  for (size_t i = 0; i < bsql_names.size(); ++i) {
    result_names.Set(i, Napi::String::New(env, bsql_names[i]));
  }

  auto result = Napi::Object::New(env);
  result.Set("names", result_names);
  result.Set("table", Table::New(env, std::move(bsql_table)));
  return result;
}

}  // namespace blazingsql
}  // namespace nv
