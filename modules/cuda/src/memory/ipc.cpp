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
#include "node_cuda/utilities/napi_to_cpp.hpp"

namespace nv {

Napi::FunctionReference IpcMemory::constructor;

Napi::Object IpcMemory::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor =
    DefineClass(env,
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
  IpcMemory::constructor = Napi::Persistent(ctor);
  IpcMemory::constructor.SuppressDestruct();

  exports.Set("IpcMemory", ctor);

  return exports;
}

IpcMemory::IpcMemory(CallbackArgs const& args) : Napi::ObjectWrap<IpcMemory>(args), Memory(args) {
  if (args.Length() == 1) { Initialize(args[0]); }
}

Napi::Object IpcMemory::New(cudaIpcMemHandle_t const& handle) {
  auto inst = IpcMemory::constructor.New({});
  IpcMemory::Unwrap(inst)->Initialize(handle);
  return inst;
}

void IpcMemory::Initialize(cudaIpcMemHandle_t const& handle) {
  NODE_CUDA_TRY(cudaIpcOpenMemHandle(&data_, handle, cudaIpcMemLazyEnablePeerAccess), Env());
  NODE_CU_TRY(cuMemGetAddressRange(nullptr, &size_, ptr()), Env());
  Napi::MemoryManagement::AdjustExternalMemory(Env(), size_);
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

Napi::Value IpcMemory::close(Napi::CallbackInfo const& info) {
  close(info.Env());
  return info.Env().Undefined();
}

Napi::Value IpcMemory::slice(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  int64_t lhs        = args.Length() > 0 ? args[0] : 0;
  int64_t rhs        = args.Length() > 1 ? args[1] : size_;
  std::tie(lhs, rhs) = clamp_slice_args(size_, lhs, rhs);
  auto copy          = DeviceMemory::New(rhs - lhs);
  if (rhs - lhs > 0) {
    NODE_CUDA_TRY(
      cudaMemcpy(DeviceMemory::Unwrap(copy)->base(), base() + lhs, rhs - lhs, cudaMemcpyDefault));
  }
  return copy;
}

Napi::FunctionReference IpcHandle::constructor;

Napi::Object IpcHandle::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor =
    DefineClass(env,
                "IpcHandle",
                {
                  InstanceValue(Napi::Symbol::WellKnown(env, "toStringTag"),
                                Napi::String::New(env, "IpcHandle"),
                                napi_enumerable),
                  InstanceAccessor("buffer", &IpcHandle::buffer, nullptr, napi_enumerable),
                  InstanceAccessor("device", &IpcHandle::device, nullptr, napi_enumerable),
                  InstanceAccessor("handle", &IpcHandle::handle, nullptr, napi_enumerable),
                  InstanceMethod("close", &IpcHandle::close),
                });
  IpcHandle::constructor = Napi::Persistent(ctor);
  IpcHandle::constructor.SuppressDestruct();

  exports.Set("IpcHandle", ctor);

  return exports;
};

IpcHandle::IpcHandle(CallbackArgs const& args) : Napi::ObjectWrap<IpcHandle>(args) {
  Initialize(args[0]);
}

Napi::Object IpcHandle::New(DeviceMemory const& dmem) {
  auto inst = IpcHandle::constructor.New({});
  IpcHandle::Unwrap(inst)->Initialize(dmem);
  return inst;
}

void IpcHandle::Initialize(DeviceMemory const& dmem) {
  dmem_.Reset(dmem.Value(), 1);
  handle_.Reset(Napi::Uint8Array::New(Env(), CUDA_IPC_HANDLE_SIZE), 1);
  NODE_CUDA_TRY(cudaIpcGetMemHandle(handle(), dmem.data()), Env());
}

void IpcHandle::Finalize(Napi::Env env) { close(env); }

Napi::Value IpcHandle::buffer(Napi::CallbackInfo const& info) { return dmem_.Value(); }

Napi::Value IpcHandle::device(Napi::CallbackInfo const& info) { return CPPToNapi(info)(device()); }

Napi::Value IpcHandle::handle(Napi::CallbackInfo const& info) { return handle_.Value(); }

void IpcHandle::close() { close(Env()); }

void IpcHandle::close(Napi::Env const& env) {
  if (!dmem_.IsEmpty()) { cudaIpcCloseMemHandle(DeviceMemory::Unwrap(dmem_.Value())->data()); }
  dmem_.Reset();
  handle_.Reset();
}

Napi::Value IpcHandle::close(Napi::CallbackInfo const& info) {
  close(info.Env());
  return info.Env().Undefined();
}

}  // namespace nv
