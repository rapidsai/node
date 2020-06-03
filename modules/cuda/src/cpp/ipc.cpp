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

#include <node_cuda/buffer.hpp>
#include <node_cuda/casting.hpp>
#include <node_cuda/macros.hpp>

namespace node_cuda {

// cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle, void *devPtr)
Napi::Value cudaIpcGetMemHandle(Napi::CallbackInfo const& info) {
  auto env   = info.Env();
  void* dptr = FromJS(info[0]);
  cudaIpcMemHandle_t handle;

  CUDA_TRY(env, CUDARTAPI::cudaIpcGetMemHandle(&handle, dptr));
  return ToNapi(env)(handle);
}

// cudaError_t cudaIpcOpenMemHandle(void **devPtr, cudaIpcMemHandle_t handle,
// unsigned int flags)
Napi::Value cudaIpcOpenMemHandle(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  void* dptr;
  size_t size;
  cudaIpcMemHandle_t handle = FromJS(info[0]);

  CUDA_TRY(env, CUDARTAPI::cudaIpcOpenMemHandle(&dptr, handle, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS));
  CU_TRY(env, CUDAAPI::cuMemGetAddressRange(nullptr, &size, reinterpret_cast<CUdeviceptr>(dptr)));

  return node_cuda::CUDABuffer::New(dptr, size, buffer_type::IPC);
}

// cudaError_t cudaIpcCloseMemHandle(void *devPtr)
Napi::Value cudaIpcCloseMemHandle(Napi::CallbackInfo const& info) {
  auto env   = info.Env();
  void* dptr = FromJS(info[0]);
  CUDA_TRY(env, CUDARTAPI::cudaIpcCloseMemHandle(dptr));
  return env.Undefined();
}

namespace ipc {
Napi::Object initModule(Napi::Env env, Napi::Object exports) {
  EXPORT_FUNC(env, exports, "getMemHandle", node_cuda::cudaIpcGetMemHandle);
  EXPORT_FUNC(env, exports, "openMemHandle", node_cuda::cudaIpcOpenMemHandle);
  EXPORT_FUNC(env, exports, "closeMemHandle", node_cuda::cudaIpcCloseMemHandle);

  return exports;
}
}  // namespace ipc
}  // namespace node_cuda
