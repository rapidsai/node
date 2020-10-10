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

#include "addon.hpp"
#include "node_cuda/device.hpp"
#include "node_cuda/macros.hpp"
#include "node_cuda/memory.hpp"
#include "node_cuda/utilities/cpp_to_napi.hpp"
#include "node_cuda/utilities/napi_to_cpp.hpp"

#include <nv_node/utilities/args.hpp>

namespace nv {

// CUresult cuInit(unsigned int Flags)
Napi::Value cuInit(CallbackArgs const& info) {
  auto env = info.Env();
  CU_TRY(env, CUDAAPI::cuInit(0));
  return info.This();
}

// CUresult cuDriverGetVersion(int* driverVersion);
Napi::Value cuDriverGetVersion(CallbackArgs const& info) {
  auto env = info.Env();
  int driverVersion;
  CU_TRY(info.Env(), CUDAAPI::cuDriverGetVersion(&driverVersion));
  return CPPToNapi(info)(driverVersion);
}

}  // namespace nv

Napi::Object initModule(Napi::Env env, Napi::Object exports) {
  EXPORT_FUNC(env, exports, "init", nv::cuInit);
  EXPORT_FUNC(env, exports, "getDriverVersion", nv::cuDriverGetVersion);

  auto gl      = Napi::Object::New(env);
  auto ipc     = Napi::Object::New(env);
  auto kernel  = Napi::Object::New(env);
  auto math    = Napi::Object::New(env);
  auto mem     = Napi::Object::New(env);
  auto program = Napi::Object::New(env);
  auto stream  = Napi::Object::New(env);
  auto texture = Napi::Object::New(env);

  EXPORT_PROP(exports, "gl", gl);
  EXPORT_PROP(exports, "ipc", ipc);
  EXPORT_PROP(exports, "kernel", kernel);
  EXPORT_PROP(exports, "math", math);
  EXPORT_PROP(exports, "mem", mem);
  EXPORT_PROP(exports, "program", program);
  EXPORT_PROP(exports, "stream", stream);
  EXPORT_PROP(exports, "texture", stream);

  nv::gl::initModule(env, gl);
  nv::ipc::initModule(env, ipc);
  nv::kernel::initModule(env, kernel);
  nv::math::initModule(env, math);
  nv::mem::initModule(env, mem);
  nv::program::initModule(env, program);
  nv::stream::initModule(env, stream);
  nv::texture::initModule(env, stream);

  EXPORT_ENUM(env, exports, "VERSION", CUDA_VERSION);
  EXPORT_ENUM(env, exports, "IPC_HANDLE_SIZE", CU_IPC_HANDLE_SIZE);

  nv::Device::Init(env, exports);
  nv::IpcMemory::Init(env, exports);
  nv::PinnedMemory::Init(env, exports);
  nv::DeviceMemory::Init(env, exports);
  nv::ManagedMemory::Init(env, exports);

  return exports;
}

NODE_API_MODULE(node_cuda, initModule);
