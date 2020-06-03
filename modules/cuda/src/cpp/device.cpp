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

#include <cuda_runtime.h>
#include <napi.h>

#include <node_cuda/casting.hpp>
#include <node_cuda/macros.hpp>
#include <node_cuda/types.hpp>

namespace node_cuda {

// cudaError_t CUDARTAPI::cudaChooseDevice(int *device, const struct
// cudaDeviceProp *prop);
Napi::Value cudaChooseDevice(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CUdevice device;
  CUDARTAPI::cudaDeviceProp props = FromJS(info[0]);
  CUDA_TRY(env, CUDARTAPI::cudaChooseDevice(&device, &props));
  // node_cuda::types::free(&props);
  return ToNapi(env)(device);
}

// cudaError_t CUDARTAPI::cudaGetDeviceCount(int *count);
Napi::Value cudaGetDeviceCount(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  int count;
  CUDA_TRY(env, CUDARTAPI::cudaGetDeviceCount(&count));
  return ToNapi(env)(count);
}

// // CUresult CUDARTAPI cuDeviceGet(CUdevice *device, int ordinal);
// Napi::Value cuDeviceGet(Napi::CallbackInfo const& info) {
//   auto env = info.Env();
//
//   int32_t ordinal = FromJS(info[0]);
//   CUdevice device;
//   CU_TRY(env, CUDARTAPI::cuDeviceGet(&device, ordinal));
//   return ToNapi(env)(device);
// }

Napi::Value cudaChooseDeviceByIndex(Napi::CallbackInfo const& info) {
  auto env        = info.Env();
  int32_t ordinal = FromJS(info[0]);
  CUdevice device;
  cudaDeviceProp props{};
  CUDA_TRY(env, CUDARTAPI::cudaGetDeviceProperties(&props, ordinal));
  CUDA_TRY(env, CUDARTAPI::cudaChooseDevice(&device, &props));
  return ToNapi(env)(device);
}

// cudaError_t CUDARTAPI cudaDeviceGetPCIBusId(char *pciBusId, int len, int
// device);
Napi::Value cudaDeviceGetPCIBusId(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  char pciBusId[256];
  CUdevice device = FromJS(info[0]);
  CUDA_TRY(env, CUDARTAPI::cudaDeviceGetPCIBusId(pciBusId, 256, device));
  return ToNapi(env)(pciBusId);
}

// cudaError_t CUDARTAPI cudaDeviceGetByPCIBusId(int *device, const char
// *pciBusId);
Napi::Value cudaDeviceGetByPCIBusId(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CUdevice device;
  std::string pciBusId = FromJS(info[0]);
  CUDA_TRY(env, CUDARTAPI::cudaDeviceGetByPCIBusId(&device, pciBusId.c_str()));
  return ToNapi(env)(device);
}

// cudaError_t CUDARTAPI cudaGetDevice(CUdevice *device);
Napi::Value cudaGetDevice(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CUdevice device;
  CUDA_TRY(env, CUDARTAPI::cudaGetDevice(&device));
  return ToNapi(env)(device);
}

// cudaError_t CUDARTAPI cudaGetDeviceFlags(unsigned int *flags);
Napi::Value cudaGetDeviceFlags(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  uint32_t flags;
  CUDA_TRY(env, CUDARTAPI::cudaGetDeviceFlags(&flags));
  return ToNapi(env)(flags);
}

// cudaError_t CUDARTAPI cudaSetDevice(CUdevice *device);
Napi::Value cudaSetDevice(Napi::CallbackInfo const& info) {
  auto env        = info.Env();
  CUdevice device = FromJS(info[0]);
  CUDA_TRY(env, CUDARTAPI::cudaSetDevice(device));
  return env.Undefined();
}

// cudaError_t CUDARTAPI cudaSetDeviceFlags(unsigned int flags);
Napi::Value cudaSetDeviceFlags(Napi::CallbackInfo const& info) {
  auto env       = info.Env();
  uint32_t flags = FromJS(info[0]);
  CUDA_TRY(env, CUDARTAPI::cudaSetDeviceFlags(flags));
  return env.Undefined();
}

// cudaError_t CUDARTAPI cudaGetDeviceProperties(struct cudaDeviceProp *prop,
// int device);
Napi::Value cudaGetDeviceProperties(Napi::CallbackInfo const& info) {
  auto env        = info.Env();
  CUdevice device = FromJS(info[0]);
  CUDARTAPI::cudaDeviceProp props{};
  CUDA_TRY(env, CUDARTAPI::cudaGetDeviceProperties(&props, device));
  auto result = ToNapi(env)(props);
  // node_cuda::types::free(&props);
  return result;
}

// cudaError_t CUDARTAPI cudaDeviceReset();
Napi::Value cudaDeviceReset(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CUDA_TRY(env, CUDARTAPI::cudaDeviceReset());
  return env.Undefined();
}

// cudaError_t CUDARTAPI cudaDeviceSynchronize(void);
Napi::Value cudaDeviceSynchronize(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CUDA_TRY(env, CUDARTAPI::cudaDeviceSynchronize());
  return env.Undefined();
}

// cudaError_t CUDARTAPI cudaDeviceCanAccessPeer(int *canAccessPeer, int device,
// int peerDevice);
Napi::Value cudaDeviceCanAccessPeer(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  int canAccessPeer;
  CUdevice device     = FromJS(info[0]);
  CUdevice peerDevice = FromJS(info[1]);
  CUDA_TRY(env, CUDARTAPI::cudaDeviceCanAccessPeer(&canAccessPeer, device, peerDevice));
  return ToNapi(env)(canAccessPeer != 0);
}

// cudaError_t CUDARTAPI cudaDeviceEnablePeerAccess(int peerDevice, unsigned int
// flags);
Napi::Value cudaDeviceEnablePeerAccess(Napi::CallbackInfo const& info) {
  auto env            = info.Env();
  CUdevice peerDevice = FromJS(info[0]);
  uint32_t flags      = FromJS(info[1]);
  CUDA_TRY(env, CUDARTAPI::cudaDeviceEnablePeerAccess(peerDevice, flags));
  return env.Undefined();
}

// cudaError_t CUDARTAPI cudaDeviceDisablePeerAccess(int peerDevice, unsigned
// int flags);
Napi::Value cudaDeviceDisablePeerAccess(Napi::CallbackInfo const& info) {
  auto env            = info.Env();
  CUdevice peerDevice = FromJS(info[0]);
  CUDA_TRY(env, CUDARTAPI::cudaDeviceDisablePeerAccess(peerDevice));
  return env.Undefined();
}

namespace device {
Napi::Object initModule(Napi::Env env, Napi::Object exports) {
  EXPORT_FUNC(env, exports, "choose", node_cuda::cudaChooseDevice);
  EXPORT_FUNC(env, exports, "getCount", node_cuda::cudaGetDeviceCount);
  // EXPORT_FUNC(env, exports, "getByIndex", node_cuda::cuDeviceGet);
  EXPORT_FUNC(env, exports, "getByIndex", node_cuda::cudaChooseDeviceByIndex);

  EXPORT_FUNC(env, exports, "getPCIBusId", node_cuda::cudaDeviceGetPCIBusId);
  EXPORT_FUNC(env, exports, "getByPCIBusId", node_cuda::cudaDeviceGetByPCIBusId);
  EXPORT_FUNC(env, exports, "get", node_cuda::cudaGetDevice);
  EXPORT_FUNC(env, exports, "getFlags", node_cuda::cudaGetDeviceFlags);
  EXPORT_FUNC(env, exports, "getProperties", node_cuda::cudaGetDeviceProperties);
  EXPORT_FUNC(env, exports, "set", node_cuda::cudaSetDevice);
  EXPORT_FUNC(env, exports, "setFlags", node_cuda::cudaSetDeviceFlags);
  EXPORT_FUNC(env, exports, "reset", node_cuda::cudaDeviceReset);
  EXPORT_FUNC(env, exports, "synchronize", node_cuda::cudaDeviceSynchronize);
  EXPORT_FUNC(env, exports, "canAccessPeer", node_cuda::cudaDeviceCanAccessPeer);
  EXPORT_FUNC(env, exports, "enablePeerAccess", node_cuda::cudaDeviceEnablePeerAccess);
  EXPORT_FUNC(env, exports, "disablePeerAccess", node_cuda::cudaDeviceDisablePeerAccess);

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
}  // namespace node_cuda
