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

#include "node_cuda/device.hpp"
#include "node_cuda/macros.hpp"
#include "node_cuda/utilities/cpp_to_napi.hpp"
#include "node_cuda/utilities/error.hpp"
#include "node_cuda/utilities/napi_to_cpp.hpp"

#include <nv_node/utilities/args.hpp>

namespace nv {

Napi::FunctionReference Device::constructor;

Napi::Value Device::get_num_devices(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(Device::get_num_devices());
}

Napi::Value Device::active_device_id(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(Device::active_device_id());
}

Napi::Object Device::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor = DefineClass(
    env,
    "Device",
    {
      StaticAccessor("numDevices", &Device::get_num_devices, nullptr, napi_enumerable),
      StaticAccessor("activeDeviceId", &Device::active_device_id, nullptr, napi_enumerable),
      InstanceAccessor("id", &Device::id, nullptr, napi_enumerable),
      InstanceAccessor("pciBusName", &Device::pci_bus_name, nullptr, napi_enumerable),
      InstanceMethod("reset", &Device::reset),
      InstanceMethod("activate", &Device::activate),
      InstanceMethod("getFlags", &Device::get_flags),
      InstanceMethod("getProperties", &Device::get_properties),
      InstanceMethod("synchronize", &Device::synchronize),
      InstanceMethod("canAccessPeerDevice", &Device::can_access_peer_device),
      InstanceMethod("enablePeerAccess", &Device::enable_peer_access),
      InstanceMethod("disablePeerAccess", &Device::disable_peer_access),
      InstanceMethod("callInDeviceContext", &Device::call_in_device_context),
    });
  Device::constructor = Napi::Persistent(ctor);
  Device::constructor.SuppressDestruct();

  auto DeviceFlags = Napi::Object::New(env);
  EXPORT_ENUM(env, DeviceFlags, "scheduleAuto", cudaDeviceScheduleAuto);
  EXPORT_ENUM(env, DeviceFlags, "scheduleSpin", cudaDeviceScheduleSpin);
  EXPORT_ENUM(env, DeviceFlags, "scheduleYield", cudaDeviceScheduleYield);
  EXPORT_ENUM(env, DeviceFlags, "scheduleBlockingSync", cudaDeviceScheduleBlockingSync);
  EXPORT_ENUM(env, DeviceFlags, "mapHost", cudaDeviceMapHost);
  EXPORT_ENUM(env, DeviceFlags, "lmemResizeToMax", cudaDeviceLmemResizeToMax);

  exports.Set("Device", ctor);
  exports.Set("DeviceFlags", DeviceFlags);

  return exports;
}

Device::Device(CallbackArgs const& args) : Napi::ObjectWrap<Device>(args) {
  NODE_CUDA_EXPECT(args.IsConstructCall(), "Device constructor requires 'new'");
  switch (args.Length()) {
    case 0: Initialize(); break;
    case 1:
      NODE_CUDA_EXPECT(args[0].IsNumber(),
                       "Device constructor requires a numeric deviceId argument");
      Initialize(args[0]);
      break;
    case 2:
      NODE_CUDA_EXPECT(args[0].IsNumber(),
                       "Device constructor requires a numeric deviceId argument");
      NODE_CUDA_EXPECT(args[1].IsNumber(),
                       "Device constructor requires a numeric CUDADeviceFlags argument");
      Initialize(args[0], args[1]);
      break;
    default:
      NODE_CUDA_EXPECT(false,
                       "Device constructor requires a numeric deviceId argument, and an optional "
                       "numeric CUDADeviceFlags argument");
      break;
  }
}

Napi::Object Device::New(int32_t id, uint32_t flags) {
  auto inst = Device::constructor.New({});
  Device::Unwrap(inst)->Initialize(id, flags);
  return inst;
}

void Device::Initialize(int32_t id, uint32_t flags) {
  id_ = id;
  char bus_id[256];
  NODE_CUDA_TRY(cudaGetDeviceProperties(&props_, id_), Env());
  NODE_CUDA_TRY(cudaDeviceGetPCIBusId(bus_id, 256, id_), Env());
  pci_bus_name_ = std::string{bus_id};
  this->reset(flags).activate();
}

Napi::Value Device::id(Napi::CallbackInfo const& info) { return CPPToNapi(info)(id()); }

Napi::Value Device::pci_bus_name(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(pci_bus_name());
}

Device& Device::reset(uint32_t flags) {
  call_in_context([&]() {
    NODE_CUDA_TRY(cudaDeviceReset(), Env());
    NODE_CUDA_TRY(cudaSetDeviceFlags(flags), Env());
    NODE_CUDA_TRY(cudaDeviceSynchronize(), Env());
  });
  return *this;
}

Napi::Value Device::reset(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  reset(args[0].IsNumber() ? args[0] : cudaDeviceScheduleAuto);
  return info.This();
}

Napi::Value Device::get_flags(Napi::CallbackInfo const& info) {
  uint32_t flags;
  call_in_context([&]() {  //
    NODE_CUDA_TRY(cudaGetDeviceFlags(&flags), Env());
  });
  return CPPToNapi(info)(flags - cudaDeviceMapHost);
}

Napi::Value Device::get_properties(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(props_);
}

Device& Device::activate() {
  if (active_device_id() != id()) { NODE_CUDA_TRY(cudaSetDevice(id()), Env()); }
  return *this;
}

Napi::Value Device::activate(Napi::CallbackInfo const& info) {
  activate();
  return info.This();
}

Device& Device::synchronize() {
  call_in_context([&]() { NODE_CUDA_TRY(cudaDeviceSynchronize(), Env()); });
  return *this;
}

Napi::Value Device::synchronize(Napi::CallbackInfo const& info) {
  synchronize();
  return info.This();
}

bool Device::can_access_peer_device(Device const& peer) const {
  int32_t can_access_peer{0};
  NODE_CUDA_TRY(cudaDeviceCanAccessPeer(&can_access_peer, id(), peer.id()), Env());
  return can_access_peer != 0;
}

Napi::Value Device::can_access_peer_device(Napi::CallbackInfo const& info) {
  Device const& peer = CallbackArgs{info}[0];
  return CPPToNapi(info)(can_access_peer_device(peer));
}

Device& Device::enable_peer_access(Device const& peer) {
  call_in_context([&]() { NODE_CUDA_TRY(cudaDeviceEnablePeerAccess(peer.id(), 0), Env()); });
  return *this;
}

Napi::Value Device::enable_peer_access(Napi::CallbackInfo const& info) {
  Device const& peer = CallbackArgs{info}[0];
  enable_peer_access(peer);
  return info.This();
}

Device& Device::disable_peer_access(Device const& peer) {
  call_in_context([&]() { NODE_CUDA_TRY(cudaDeviceDisablePeerAccess(peer.id()), Env()); });
  return *this;
}

Napi::Value Device::disable_peer_access(Napi::CallbackInfo const& info) {
  Device const& peer = CallbackArgs{info}[0];
  disable_peer_access(peer);
  return info.This();
}

Napi::Value Device::call_in_device_context(Napi::CallbackInfo const& info) {
  if (info.Length() == 1 and info[0].IsFunction()) {
    auto callback = info[0].As<Napi::Function>();
    call_in_context([&] { callback({}); });
  }
  return info.This();
}

}  // namespace nv
