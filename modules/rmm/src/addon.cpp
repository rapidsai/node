// Copyright (c) 2020, NVIDIA CORPORATION.
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

#include "node_rmm/addon.hpp"
#include "node_rmm/device_buffer.hpp"
#include "node_rmm/memory_resource.hpp"

#include <nv_node/macros.hpp>

namespace nv {
Napi::Value rmmInit(Napi::CallbackInfo const& info) {
  // todo
  return info.This();
}

Napi::Value set_per_device_resource(CallbackArgs const& args) {
  rmm::mr::set_per_device_resource(args[0], args[1]);
  return args.Env().Undefined();
}

}  // namespace nv

Napi::Object initModule(Napi::Env env, Napi::Object exports) {
  EXPORT_FUNC(env, exports, "init", nv::rmmInit);
  EXPORT_FUNC(env, exports, "setPerDeviceResource", nv::set_per_device_resource);
  nv::MemoryResource::Init(env, exports);
  nv::DeviceBuffer::Init(env, exports);

  // Create a persistent reference to the exports object as the add-on instance data.
  // This will allow this add-on to support multiple instances of itself running on multiple worker
  // threads, as well as multiple instances of itself running in different contexts on the same
  // thread.
  env.SetInstanceData<Napi::ObjectReference>(new Napi::ObjectReference(Napi::Persistent(exports)));

  return exports;
}

NODE_API_MODULE(node_rmm, initModule);
