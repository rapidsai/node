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
#include "cache.hpp"

#include <nv_node/utilities/args.hpp>

#include <engine/engine.h>
#include <engine/initialize.h>

namespace nv {

Napi::Function Context::Init(Napi::Env env, Napi::Object exports) {
  return DefineClass(env, "Context", {});
}

Context::Context(Napi::CallbackInfo const& info) : EnvLocalObjectWrap<Context>(info) {
  auto env = info.Env();

  NapiToCPP::Object props                       = info[0];
  uint16_t ralId                                = props.Get("ralId");
  std::string worker_id                         = props.Get("workerId");
  std::string network_iface_name                = props.Get("network_iface_name");
  int32_t ralCommunicationPort                  = props.Get("ralCommunicationPort");
  std::vector<NodeMetaDataUCP> workers_ucp_info = props.Get("workersUcpInfo");
  bool singleNode                               = props.Get("singleNode");
  std::string allocation_mode                   = props.Get("allocationMode");
  std::size_t initial_pool_size                 = props.Get("initialPoolSize");
  std::size_t maximum_pool_size                 = props.Get("maximumPoolSize");
  bool enable_logging                           = props.Get("enableLogging");

  auto config_options = [&] {
    std::map<std::string, std::string> config{};
    auto prop = props.Get("configOptions");
    if (prop.IsObject() and not prop.IsNull()) {
      auto opts = prop.As<Napi::Object>();
      auto keys = opts.GetPropertyNames();
      for (auto i = 0u; i < keys.Length(); ++i) {
        Napi::HandleScope scope(env);
        auto name    = keys.Get(i).ToString();
        config[name] = opts.Get(name).ToString();
      }
    }
    return config;
  }();

  auto init_result = ::initialize(ralId,
                                  worker_id,
                                  network_iface_name,
                                  ralCommunicationPort,
                                  workers_ucp_info,
                                  singleNode,
                                  config_options,
                                  allocation_mode,
                                  initial_pool_size,
                                  maximum_pool_size,
                                  enable_logging);
  auto& caches     = init_result.first;
  _port            = init_result.second;
  _transport_out   = Napi::Persistent(CacheMachine::New(env, caches.first));
  _transport_in    = Napi::Persistent(CacheMachine::New(env, caches.second));
}

}  // namespace nv
