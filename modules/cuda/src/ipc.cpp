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

#include "buffer.hpp"
#include "node_cuda/macros.hpp"
#include "node_cuda/utilities/cpp_to_napi.hpp"
#include "node_cuda/utilities/napi_to_cpp.hpp"

#include <cuda_runtime_api.h>
#include <nv_node/utilities/args.hpp>

namespace nv {

// cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle, void *devPtr)
Napi::Value cudaIpcGetMemHandle(CallbackArgs const& info) {
  auto env = info.Env();
  cudaIpcMemHandle_t handle;

  CUDA_TRY(env, CUDARTAPI::cudaIpcGetMemHandle(&handle, info[0]));
  return CPPToNapi(info)(handle);
}

// cudaError_t cudaIpcOpenMemHandle(void **devPtr, cudaIpcMemHandle_t handle,
// unsigned int flags)
Napi::Value cudaIpcOpenMemHandle(CallbackArgs const& info) {
  auto env = info.Env();
  void* dptr;
  size_t size;
  cudaIpcMemHandle_t const& handle = info[0];

  CUDA_TRY(env, CUDARTAPI::cudaIpcOpenMemHandle(&dptr, handle, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS));
  CU_TRY(env, CUDAAPI::cuMemGetAddressRange(nullptr, &size, reinterpret_cast<CUdeviceptr>(dptr)));

  return nv::CUDABuffer::New(dptr, size, buffer_type::IPC);
}

// cudaError_t cudaIpcCloseMemHandle(void *devPtr)
Napi::Value cudaIpcCloseMemHandle(CallbackArgs const& info) {
  auto env = info.Env();
  CUDA_TRY(env, CUDARTAPI::cudaIpcCloseMemHandle(info[0]));
  return env.Undefined();
}

namespace ipc {
Napi::Object initModule(Napi::Env env, Napi::Object exports) {
  EXPORT_FUNC(env, exports, "getMemHandle", nv::cudaIpcGetMemHandle);
  EXPORT_FUNC(env, exports, "openMemHandle", nv::cudaIpcOpenMemHandle);
  EXPORT_FUNC(env, exports, "closeMemHandle", nv::cudaIpcCloseMemHandle);

  return exports;
}
}  // namespace ipc
}  // namespace nv
