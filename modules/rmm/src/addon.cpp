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

#include "node_rmm/device_buffer.hpp"
#include "node_rmm/memory_resource.hpp"

#include <node_cuda/device.hpp>
#include <node_cuda/utilities/error.hpp>

#include <nv_node/addon.hpp>
#include <nv_node/macros.hpp>

struct node_rmm : public nv::EnvLocalAddon, public Napi::Addon<node_rmm> {
  node_rmm(Napi::Env const& env, Napi::Object exports) : EnvLocalAddon(env, exports) {
    auto const num_devices = nv::Device::get_num_devices();
    _per_device_resources  = Napi::Persistent(Napi::Array::New(env, num_devices));
    _after_init = Napi::Persistent(Napi::Function::New(env, [=](Napi::CallbackInfo const& info) {
      auto pdmr = _per_device_resources.Value();
      for (int32_t id = 0; id < num_devices; ++id) {
        pdmr.Set(id, nv::MemoryResource::Device(info.Env(), rmm::cuda_device_id{id}));
      }
    }));
    DefineAddon(
      exports,
      {
        InstanceMethod("init", &node_rmm::InitAddon),
        InstanceValue("_cpp_exports", _cpp_exports.Value()),
        InstanceValue("DeviceBuffer", InitClass<nv::DeviceBuffer>(env, exports)),
        InstanceValue("MemoryResource", InitClass<nv::MemoryResource>(env, exports)),
        InstanceMethod<&node_rmm::get_per_device_resource>("getPerDeviceResource"),
        InstanceMethod<&node_rmm::set_per_device_resource>("setPerDeviceResource"),
        InstanceMethod<&node_rmm::get_current_device_resource>("getCurrentDeviceResource"),
        InstanceMethod<&node_rmm::set_current_device_resource>("setCurrentDeviceResource"),
      });
  }

 private:
  Napi::Reference<Napi::Array> _per_device_resources;
  Napi::Value get_per_device_resource(Napi::CallbackInfo const& info) {
    auto device_id = info[0].ToNumber();
    NODE_CUDA_EXPECT(device_id.Int32Value() < nv::Device::get_num_devices(),
                     "getPerDeviceResource requires device_id to be less than Device.numDevices",
                     info.Env());
    return _per_device_resources.Value().Get(device_id);
  }
  Napi::Value set_per_device_resource(Napi::CallbackInfo const& info) {
    auto device_id = info[0].ToNumber();
    NODE_CUDA_EXPECT(device_id.Int32Value() < nv::Device::get_num_devices(),
                     "setPerDeviceResource requires device_id to be less than Device.numDevices",
                     info.Env());
    NODE_CUDA_EXPECT(nv::MemoryResource::IsInstance(info[1]),
                     "setPerDeviceResource requires a MemoryResource instance",
                     info.Env());
    auto prev = _per_device_resources.Value().Get(device_id);
    auto next = nv::MemoryResource::wrapper_t(info[1].ToObject());
    _per_device_resources.Value().Set(device_id, next);
    rmm::mr::set_per_device_resource(rmm::cuda_device_id{device_id},
                                     next->operator rmm::mr::device_memory_resource*());
    return prev;
  }
  Napi::Value get_current_device_resource(Napi::CallbackInfo const& info) {
    return _per_device_resources.Value().Get(nv::Device::active_device_id());
  }
  Napi::Value set_current_device_resource(Napi::CallbackInfo const& info) {
    NODE_CUDA_EXPECT(nv::MemoryResource::IsInstance(info[0]),
                     "setCurrentDeviceResource requires a MemoryResource instance",
                     info.Env());
    auto device_id = nv::Device::active_device_id();
    auto prev      = _per_device_resources.Value().Get(device_id);
    auto next      = nv::MemoryResource::wrapper_t(info[0].ToObject());
    _per_device_resources.Value().Set(device_id, next);
    rmm::mr::set_per_device_resource(rmm::cuda_device_id{device_id},
                                     next->operator rmm::mr::device_memory_resource*());
    return prev;
  }
};

NODE_API_ADDON(node_rmm);
