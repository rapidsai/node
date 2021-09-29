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
#include "nv_node/utilities/napi_to_cpp.hpp"

#include <node_cudf/table.hpp>

#include <cudf/copying.hpp>

#include <execution_graph/Context.h>

namespace nv {
namespace blazingsql {

namespace {

std::pair<std::vector<std::string>, Table::wrapper_t> get_names_and_table(NapiToCPP::Object df) {
  std::vector<std::string> names = df.Get("names");
  Napi::Function asTable         = df.Get("asTable");
  Table::wrapper_t table         = asTable.Call(df.val, {}).ToObject();
  return {std::move(names), std::move(table)};
}

}  // namespace

Napi::Function Context::Init(Napi::Env const& env, Napi::Object exports) {
  return DefineClass(env,
                     "Context",
                     {
                       InstanceMethod<&Context::send>("send"),
                       InstanceMethod<&Context::pull>("pull"),
                       InstanceMethod<&Context::broadcast>("broadcast"),
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
  Napi::Array schemas                  = args[1];
  std::vector<std::string> table_names = args[2];
  std::vector<std::string> table_scans = args[3];
  int32_t ctx_token                    = args[4];
  std::string query                    = args[5];
  auto config_opts_                    = args[6];
  std::string sql                      = args[7];
  std::string current_timestamp        = args[8];
  auto config_options                  = [&] {
    std::map<std::string, std::string> config{};
    if (!config_opts_.IsNull() && config_opts_.IsObject()) {
      auto opts = config_opts_.As<Napi::Object>();
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
    NapiToCPP::Object df = data_frames.Get(i);
    auto [names, table]  = get_names_and_table(df);

    tables.Set(i, table);
    table_views.push_back(*table);
    column_names.push_back(names);
  }
  
  for (std::size_t i = 0; i < schemas.Length(); ++i) {
    NapiToCPP::Object schema = schemas.Get(i);
    std::vector<std::string> names = schema.Get("names");

    column_names.push_back(names);
  }

  std::vector<std::string> worker_ids;
  worker_ids.reserve(_worker_ids.size());
  std::transform(
    _worker_ids.begin(), _worker_ids.end(), std::back_inserter(worker_ids), [](int32_t const id) {
      return std::to_string(id);
    });

  return blazingsql::run_generate_graph(env,
                                        *this,
                                        0,
                                        worker_ids,
                                        table_views,
                                        schemas,
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

SQLTask* Context::pull(std::string const& message_id) {
  return this->_transport_in.Value()->pull_from_cache(message_id);
}

void Context::send(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};

  int32_t dst_ral_id     = args[0];
  std::string ctx_token  = args[1];
  std::string message_id = args[2];
  NapiToCPP::Object df   = args[3];

  auto [names, table] = get_names_and_table(df);

  this->send(dst_ral_id, ctx_token, message_id, names, *table);
}

Napi::Value Context::pull(Napi::CallbackInfo const& info) {
  return pull(info[0].ToString())->run();
}

Napi::Value Context::broadcast(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  int32_t ctx_token    = args[0];
  NapiToCPP::Object df = args[1];

  Table::wrapper_t table;
  std::vector<std::string> names;
  std::tie(names, table) = get_names_and_table(df);

  auto const num_rows    = table->num_rows();
  auto const num_workers = _worker_ids.size();
  auto const num_slice_rows =
    static_cast<cudf::size_type>(ceil(static_cast<double>(num_rows) / num_workers));

  auto slices = cudf::slice(*table, [&]() {  //
    cudf::size_type count{0};
    std::vector<cudf::size_type> indices;
    std::generate_n(std::back_inserter(indices), num_workers * 2, [&]() mutable {
      return std::min(num_rows, num_slice_rows * (++count / 2));
    });
    return indices;
  }());

  auto messages = Napi::Array::New(env, num_workers);

  for (int32_t i = num_workers; --i > -1;) {
    auto const id  = _worker_ids[i];
    auto const tok = std::to_string(ctx_token + i);
    auto const msg = "broadcast_table_message_" + tok;

    messages[i] = msg;

    if (id != _id) {
      this->send(id, tok, msg, names, slices[i]);
    } else {
      this->_transport_in.Value()->add_to_cache(
        get_node_id(), get_ral_id(), get_ral_id(), tok, msg, names, slices[i]);
    }
  }

  return messages;
}

}  // namespace blazingsql
}  // namespace nv
