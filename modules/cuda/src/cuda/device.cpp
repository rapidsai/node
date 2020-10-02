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

#include "cuda/device.hpp"
#include "cuda/utilities/cpp_to_napi.hpp"
#include "cuda/utilities/error.hpp"

#include <cuda_runtime_api.h>
#include <napi.h>
#include <exception>
#include <functional>
#include <nv_node/utilities/args.hpp>

namespace nv {

Napi::FunctionReference Device::constructor;

Napi::Object Device::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor =
    DefineClass(env,
                "Device",
                {
                  InstanceAccessor("id", &Device::GetId, nullptr, napi_enumerable),
                  InstanceAccessor("name", &Device::GetName, nullptr, napi_enumerable),
                  InstanceAccessor("pciBusId", &Device::GetPCIBusId, nullptr, napi_enumerable),
                  InstanceAccessor("pciBusName", &Device::GetPCIBusName, nullptr, napi_enumerable),
                  InstanceMethod("reset", &Device::reset),
                  InstanceMethod("activate", &Device::activate),
                  InstanceMethod("synchronize", &Device::synchronize),
                  InstanceMethod("canAccessPeerDevice", &Device::can_access_peer_device),
                  InstanceMethod("enablePeerAccess", &Device::enable_peer_access),
                  InstanceMethod("disablePeerAccess", &Device::disable_peer_access),
                });
  Device::constructor = Napi::Persistent(ctor);
  Device::constructor.SuppressDestruct();

  exports.Set("Device", ctor);

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
      Initialize(args[0], args[2]);
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
  Device::Unwrap(inst)->Initialize(id);
  return inst;
}

void Device::Initialize(int32_t id, uint32_t flags) {
  id_ = id;
  char bus_id[256];
  NODE_CUDA_TRY(cudaGetDeviceProperties(&props_, id_), Env());
  NODE_CUDA_TRY(cudaDeviceGetPCIBusId(bus_id, 256, id_), Env());
  pci_bus_name_ = std::string{bus_id};
}

Napi::Value Device::GetId(Napi::CallbackInfo const& info) { return CPPToNapi(info)(id()); }

Napi::Value Device::GetName(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(std::string{props().name});
}

Napi::Value Device::GetPCIBusId(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(props().pciBusID);
}

Napi::Value Device::GetPCIBusName(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(pci_bus_name());
}

Device const& Device::reset(uint32_t flags) {
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

Device const& Device::activate() {
  if (current_device_id() != id()) {  //
    NODE_CUDA_TRY(cudaSetDevice(id()), Env());
  }
  return *this;
}

Napi::Value Device::activate(Napi::CallbackInfo const& info) {
  activate();
  return info.This();
}

Device const& Device::synchronize() {
  call_in_context([&]() {  //
    NODE_CUDA_TRY(cudaDeviceSynchronize(), Env());
  });
  return *this;
}

Napi::Value Device::synchronize(Napi::CallbackInfo const& info) {
  synchronize();
  return info.This();
}

bool Device::can_access_peer_device(Device const& peer) {
  int32_t can_access_peer{0};
  NODE_CUDA_TRY(cudaDeviceCanAccessPeer(&can_access_peer, id(), peer.id()), Env());
  return can_access_peer != 0;
}

Napi::Value Device::can_access_peer_device(Napi::CallbackInfo const& info) {
  can_access_peer_device(CallbackArgs{info}[0].operator Device());
  return info.This();
}

Device const& Device::enable_peer_access(Device const& peer) {
  call_in_context([&]() { NODE_CUDA_TRY(cudaDeviceEnablePeerAccess(peer.id(), 0), Env()); });
  return *this;
}

Napi::Value Device::enable_peer_access(Napi::CallbackInfo const& info) {
  enable_peer_access(CallbackArgs{info}[0].operator Device());
  return info.This();
}

Device const& Device::disable_peer_access(Device const& peer) {
  call_in_context([&]() { NODE_CUDA_TRY(cudaDeviceDisablePeerAccess(peer.id()), Env()); });
  return *this;
}

Napi::Value Device::disable_peer_access(Napi::CallbackInfo const& info) {
  disable_peer_access(CallbackArgs{info}[0].operator Device());
  return info.This();
}

}  // namespace nv
