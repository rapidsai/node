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

#include "node_cuda/memory.hpp"
#include "node_cuda/utilities/error.hpp"

#include <cuda_runtime_api.h>
#include <napi.h>

namespace nv {

Napi::FunctionReference IpcMemory::constructor;

Napi::Object IpcMemory::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor =
    DefineClass(env,
                "IPCMemory",
                {
                  InstanceAccessor("byteLength", &IpcMemory::size, nullptr, napi_enumerable),
                  InstanceAccessor("device", &IpcMemory::device, nullptr, napi_enumerable),
                  InstanceAccessor("ptr", &IpcMemory::ptr, nullptr, napi_enumerable),
                  InstanceMethod("slice", &IpcMemory::slice),
                  InstanceMethod("close", &IpcMemory::close_handle),
                });
  IpcMemory::constructor = Napi::Persistent(ctor);
  IpcMemory::constructor.SuppressDestruct();

  exports.Set("IPCMemory", ctor);

  return exports;
}

IpcMemory::IpcMemory(CallbackArgs const& args) : Napi::ObjectWrap<IpcMemory>(args), Memory(args) {
  if (args.Length() == 1) { Initialize(*static_cast<cudaIpcMemHandle_t*>(args[0])); }
}

Napi::Object IpcMemory::New(cudaIpcMemHandle_t const& handle) {
  auto inst = IpcMemory::constructor.New({});
  IpcMemory::Unwrap(inst)->Initialize(handle);
  return inst;
}

void IpcMemory::Initialize(cudaIpcMemHandle_t const& handle) {
  NODE_CUDA_TRY(cudaIpcOpenMemHandle(&data_, handle, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS), Env());
  NODE_CU_TRY(cuMemGetAddressRange(nullptr, &size_, ptr()), Env());
  Napi::MemoryManagement::AdjustExternalMemory(Env(), size_);
}

void IpcMemory::Finalize(Napi::Env env) {}

void IpcMemory::close_handle() { close_handle(Env()); }

void IpcMemory::close_handle(Napi::Env const& env) {
  if (data_ != nullptr && size_ > 0) {
    if (cudaIpcCloseMemHandle(data_) == cudaSuccess) {
      Napi::MemoryManagement::AdjustExternalMemory(env, -size_);
    }
  }
}

Napi::Value IpcMemory::close_handle(Napi::CallbackInfo const& info) {
  close_handle(info.Env());
  return info.Env().Undefined();
}

Napi::Value IpcMemory::slice(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  int64_t offset = args[0];
  int64_t size   = size_ - offset;
  if (args.Length() == 2 && args[1].IsNumber()) { size = args[1].operator int64_t() - offset; }
  auto copy = DeviceMemory::New(size = std::max<int64_t>(size, 0));
  if (size > 0) {
    NODE_CUDA_TRY(
      cudaMemcpy(DeviceMemory::Unwrap(copy)->base(), base() + offset, size, cudaMemcpyDefault));
  }
  return copy;
}

}  // namespace nv
