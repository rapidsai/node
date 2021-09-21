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

#include "node_cuda/memory.hpp"
#include "node_cuda/utilities/napi_to_cpp.hpp"

namespace nv {

Napi::Function IpcMemory::Init(Napi::Env const& env, Napi::Object exports) {
  return DefineClass(env,
                     "IPCMemory",
                     {
                       InstanceValue(Napi::Symbol::WellKnown(env, "toStringTag"),
                                     Napi::String::New(env, "IPCMemory"),
                                     napi_enumerable),
                       InstanceAccessor("byteLength", &IpcMemory::size, nullptr, napi_enumerable),
                       InstanceAccessor("device", &IpcMemory::device, nullptr, napi_enumerable),
                       InstanceAccessor("ptr", &IpcMemory::ptr, nullptr, napi_enumerable),
                       InstanceMethod("slice", &IpcMemory::slice),
                       InstanceMethod("close", &IpcMemory::close),
                     });
}

IpcMemory::IpcMemory(CallbackArgs const& args) : EnvLocalObjectWrap<IpcMemory>(args), Memory(args) {
  if (args.Length() == 1) {
    cudaIpcMemHandle_t const handle = args[0];
    NODE_CUDA_TRY(cudaIpcOpenMemHandle(&data_, handle, cudaIpcMemLazyEnablePeerAccess), Env());
    NODE_CU_TRY(cuMemGetAddressRange(nullptr, &size_, ptr()), Env());
    Napi::MemoryManagement::AdjustExternalMemory(Env(), size_);
  }
}

IpcMemory::wrapper_t IpcMemory::New(Napi::Env const& env, cudaIpcMemHandle_t const& handle) {
  return EnvLocalObjectWrap<IpcMemory>::New(
    env, {Napi::External<cudaIpcMemHandle_t>::New(env, const_cast<cudaIpcMemHandle_t*>(&handle))});
}

void IpcMemory::Finalize(Napi::Env env) { close(env); }

void IpcMemory::close() { close(Env()); }

void IpcMemory::close(Napi::Env const& env) {
  if (data_ != nullptr && size_ > 0) {
    if (cudaIpcCloseMemHandle(data_) == cudaSuccess) {
      Napi::MemoryManagement::AdjustExternalMemory(env, -size_);
    }
  }
  data_ = nullptr;
  size_ = 0;
}

void IpcMemory::close(Napi::CallbackInfo const& info) { close(info.Env()); }

Napi::Value IpcMemory::slice(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  int64_t lhs        = args.Length() > 0 ? args[0] : 0;
  int64_t rhs        = args.Length() > 1 ? args[1] : size_;
  std::tie(lhs, rhs) = clamp_slice_args(size_, lhs, rhs);
  auto copy          = DeviceMemory::New(info.Env(), rhs - lhs);
  if (rhs - lhs > 0) {
    NODE_CUDA_TRY(cudaMemcpy(copy->base(), base() + lhs, rhs - lhs, cudaMemcpyDefault));
  }
  return copy;
}

Napi::Function IpcHandle::Init(Napi::Env const& env, Napi::Object exports) {
  return DefineClass(env,
                     "IpcHandle",
                     {
                       InstanceValue(Napi::Symbol::WellKnown(env, "toStringTag"),
                                     Napi::String::New(env, "IpcHandle"),
                                     napi_enumerable),
                       InstanceAccessor("buffer", &IpcHandle::buffer, nullptr, napi_enumerable),
                       InstanceAccessor("device", &IpcHandle::device, nullptr, napi_enumerable),
                       InstanceAccessor("handle", &IpcHandle::handle, nullptr, napi_enumerable),
                     });
};

IpcHandle::IpcHandle(CallbackArgs const& args) : EnvLocalObjectWrap<IpcHandle>(args) {
  DeviceMemory::wrapper_t dmem = args[0].ToObject();
  dmem_                        = Napi::Persistent(dmem);
  handle_                      = Napi::Persistent(dmem->getIpcMemHandle());
}

IpcHandle::wrapper_t IpcHandle::New(Napi::Env const& env, DeviceMemory const& dmem) {
  return EnvLocalObjectWrap<IpcHandle>::New(env, dmem.Value());
}

Napi::Value IpcHandle::buffer(Napi::CallbackInfo const& info) { return dmem_.Value(); }

Napi::Value IpcHandle::device(Napi::CallbackInfo const& info) { return CPPToNapi(info)(device()); }

Napi::Value IpcHandle::handle(Napi::CallbackInfo const& info) { return handle_.Value(); }

}  // namespace nv
