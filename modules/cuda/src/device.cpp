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

#include "cuda/utilities/cpp_to_napi.hpp"
#include "cuda/utilities/napi_to_cpp.hpp"
#include "macros.hpp"
#include "types.hpp"

#include <cuda_runtime_api.h>
#include <napi.h>
#include <nv_node/utilities/args.hpp>

namespace nv {

// cudaError_t CUDARTAPI::cudaChooseDevice(int *device, const struct
// cudaDeviceProp *prop);
Napi::Value cudaChooseDevice(CallbackArgs const& info) {
  auto env = info.Env();
  CUdevice device;
  CUDARTAPI::cudaDeviceProp props = info[0];
  CUDA_TRY(env, CUDARTAPI::cudaChooseDevice(&device, &props));
  // nv::types::free(&props);
  return CPPToNapi(info)(device);
}

// cudaError_t CUDARTAPI::cudaGetDeviceCount(int *count);
Napi::Value cudaGetDeviceCount(CallbackArgs const& info) {
  auto env = info.Env();
  int count;
  CUDA_TRY(env, CUDARTAPI::cudaGetDeviceCount(&count));
  return CPPToNapi(info)(count);
}

Napi::Value cudaChooseDeviceByIndex(CallbackArgs const& info) {
  auto env        = info.Env();
  int32_t ordinal = info[0];
  CUdevice device;
  cudaDeviceProp props{};
  CUDA_TRY(env, CUDARTAPI::cudaGetDeviceProperties(&props, ordinal));
  CUDA_TRY(env, CUDARTAPI::cudaChooseDevice(&device, &props));
  return CPPToNapi(info)(device);
}

// cudaError_t CUDARTAPI cudaDeviceGetPCIBusId(char *pciBusId, int len, int
// device);
Napi::Value cudaDeviceGetPCIBusId(CallbackArgs const& info) {
  auto env = info.Env();
  char pciBusId[256];
  CUdevice device = info[0];
  CUDA_TRY(env, CUDARTAPI::cudaDeviceGetPCIBusId(pciBusId, 256, device));
  return CPPToNapi(info)(pciBusId, sizeof(pciBusId));
}

// cudaError_t CUDARTAPI cudaDeviceGetByPCIBusId(int *device, const char
// *pciBusId);
Napi::Value cudaDeviceGetByPCIBusId(CallbackArgs const& info) {
  auto env = info.Env();
  CUdevice device;
  std::string pciBusId = info[0];
  CUDA_TRY(env, CUDARTAPI::cudaDeviceGetByPCIBusId(&device, pciBusId.c_str()));
  return CPPToNapi(info)(device);
}

// cudaError_t CUDARTAPI cudaGetDevice(CUdevice *device);
Napi::Value cudaGetDevice(CallbackArgs const& info) {
  auto env = info.Env();
  CUdevice device;
  CUDA_TRY(env, CUDARTAPI::cudaGetDevice(&device));
  return CPPToNapi(info)(device);
}

// cudaError_t CUDARTAPI cudaGetDeviceFlags(unsigned int *flags);
Napi::Value cudaGetDeviceFlags(CallbackArgs const& info) {
  auto env = info.Env();
  uint32_t flags;
  CUDA_TRY(env, CUDARTAPI::cudaGetDeviceFlags(&flags));
  return CPPToNapi(info)(flags);
}

// cudaError_t CUDARTAPI cudaSetDevice(CUdevice *device);
Napi::Value cudaSetDevice(CallbackArgs const& info) {
  auto env        = info.Env();
  CUdevice device = info[0];
  CUDA_TRY(env, CUDARTAPI::cudaSetDevice(device));
  return env.Undefined();
}

// cudaError_t CUDARTAPI cudaSetDeviceFlags(unsigned int flags);
Napi::Value cudaSetDeviceFlags(CallbackArgs const& info) {
  auto env       = info.Env();
  uint32_t flags = info[0];
  CUDA_TRY(env, CUDARTAPI::cudaSetDeviceFlags(flags));
  return env.Undefined();
}

// cudaError_t CUDARTAPI cudaGetDeviceProperties(struct cudaDeviceProp *prop,
// int device);
Napi::Value cudaGetDeviceProperties(CallbackArgs const& info) {
  auto env        = info.Env();
  CUdevice device = info[0];
  CUDARTAPI::cudaDeviceProp props{};
  CUDA_TRY(env, CUDARTAPI::cudaGetDeviceProperties(&props, device));
  return CPPToNapi(info)(props);
}

// cudaError_t CUDARTAPI cudaDeviceReset();
Napi::Value cudaDeviceReset(CallbackArgs const& info) {
  auto env = info.Env();
  CUDA_TRY(env, CUDARTAPI::cudaDeviceReset());
  return env.Undefined();
}

// cudaError_t CUDARTAPI cudaDeviceSynchronize(void);
Napi::Value cudaDeviceSynchronize(CallbackArgs const& info) {
  auto env = info.Env();
  CUDA_TRY(env, CUDARTAPI::cudaDeviceSynchronize());
  return env.Undefined();
}

// cudaError_t CUDARTAPI cudaDeviceCanAccessPeer(int *canAccessPeer, int device,
// int peerDevice);
Napi::Value cudaDeviceCanAccessPeer(CallbackArgs const& info) {
  auto env = info.Env();
  int canAccessPeer;
  CUdevice device     = info[0];
  CUdevice peerDevice = info[1];
  CUDA_TRY(env, CUDARTAPI::cudaDeviceCanAccessPeer(&canAccessPeer, device, peerDevice));
  return CPPToNapi(info)(canAccessPeer != 0);
}

// cudaError_t CUDARTAPI cudaDeviceEnablePeerAccess(int peerDevice, unsigned int
// flags);
Napi::Value cudaDeviceEnablePeerAccess(CallbackArgs const& info) {
  auto env            = info.Env();
  CUdevice peerDevice = info[0];
  uint32_t flags      = info[1];
  CUDA_TRY(env, CUDARTAPI::cudaDeviceEnablePeerAccess(peerDevice, flags));
  return env.Undefined();
}

// cudaError_t CUDARTAPI cudaDeviceDisablePeerAccess(int peerDevice, unsigned
// int flags);
Napi::Value cudaDeviceDisablePeerAccess(CallbackArgs const& info) {
  auto env            = info.Env();
  CUdevice peerDevice = info[0];
  CUDA_TRY(env, CUDARTAPI::cudaDeviceDisablePeerAccess(peerDevice));
  return env.Undefined();
}

namespace device {
Napi::Object initModule(Napi::Env env, Napi::Object exports) {
  EXPORT_FUNC(env, exports, "choose", nv::cudaChooseDevice);
  EXPORT_FUNC(env, exports, "getCount", nv::cudaGetDeviceCount);
  EXPORT_FUNC(env, exports, "getByIndex", nv::cudaChooseDeviceByIndex);

  EXPORT_FUNC(env, exports, "getPCIBusId", nv::cudaDeviceGetPCIBusId);
  EXPORT_FUNC(env, exports, "getByPCIBusId", nv::cudaDeviceGetByPCIBusId);
  EXPORT_FUNC(env, exports, "get", nv::cudaGetDevice);
  EXPORT_FUNC(env, exports, "getFlags", nv::cudaGetDeviceFlags);
  EXPORT_FUNC(env, exports, "getProperties", nv::cudaGetDeviceProperties);
  EXPORT_FUNC(env, exports, "set", nv::cudaSetDevice);
  EXPORT_FUNC(env, exports, "setFlags", nv::cudaSetDeviceFlags);
  EXPORT_FUNC(env, exports, "reset", nv::cudaDeviceReset);
  EXPORT_FUNC(env, exports, "synchronize", nv::cudaDeviceSynchronize);
  EXPORT_FUNC(env, exports, "canAccessPeer", nv::cudaDeviceCanAccessPeer);
  EXPORT_FUNC(env, exports, "enablePeerAccess", nv::cudaDeviceEnablePeerAccess);
  EXPORT_FUNC(env, exports, "disablePeerAccess", nv::cudaDeviceDisablePeerAccess);

  auto cudaDeviceFlags = Napi::Object::New(env);
  EXPORT_ENUM(env, cudaDeviceFlags, "scheduleAuto", cudaDeviceScheduleAuto);
  EXPORT_ENUM(env, cudaDeviceFlags, "scheduleSpin", cudaDeviceScheduleSpin);
  EXPORT_ENUM(env, cudaDeviceFlags, "scheduleYield", cudaDeviceScheduleYield);
  EXPORT_ENUM(env, cudaDeviceFlags, "scheduleBlockingSync", cudaDeviceScheduleBlockingSync);
  EXPORT_ENUM(env, cudaDeviceFlags, "mapHost", cudaDeviceMapHost);
  EXPORT_ENUM(env, cudaDeviceFlags, "lmemResizeToMax", cudaDeviceLmemResizeToMax);
  EXPORT_PROP(exports, "flags", cudaDeviceFlags);

  return exports;
}
}  // namespace device
}  // namespace nv
