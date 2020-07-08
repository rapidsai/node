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

#include <node_cuda/addon.hpp>
#include <node_cuda/casting.hpp>
#include <node_cuda/macros.hpp>

namespace node_cuda {

// CUresult cuInit(unsigned int Flags)
Napi::Value cuInit(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CU_TRY(env, CUDAAPI::cuInit(0));
  return info.This();
}

// CUresult cuDriverGetVersion(int* driverVersion);
Napi::Value cuDriverGetVersion(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  int driverVersion;
  CU_TRY(info.Env(), CUDAAPI::cuDriverGetVersion(&driverVersion));
  return node_cuda::ToNapi(info.Env())(driverVersion);
}

}  // namespace node_cuda

Napi::Object initModule(Napi::Env env, Napi::Object exports) {
  EXPORT_FUNC(env, exports, "init", node_cuda::cuInit);
  EXPORT_FUNC(env, exports, "getDriverVersion", node_cuda::cuDriverGetVersion);

  auto device  = Napi::Object::New(env);
  auto gl      = Napi::Object::New(env);
  auto ipc     = Napi::Object::New(env);
  auto kernel  = Napi::Object::New(env);
  auto math    = Napi::Object::New(env);
  auto mem     = Napi::Object::New(env);
  auto program = Napi::Object::New(env);
  auto stream  = Napi::Object::New(env);

  EXPORT_PROP(exports, "device", device);
  EXPORT_PROP(exports, "gl", gl);
  EXPORT_PROP(exports, "ipc", ipc);
  EXPORT_PROP(exports, "kernel", kernel);
  EXPORT_PROP(exports, "math", math);
  EXPORT_PROP(exports, "mem", mem);
  EXPORT_PROP(exports, "program", program);
  EXPORT_PROP(exports, "stream", stream);

  node_cuda::device::initModule(env, device);
  node_cuda::gl::initModule(env, gl);
  node_cuda::ipc::initModule(env, ipc);
  node_cuda::kernel::initModule(env, kernel);
  node_cuda::math::initModule(env, math);
  node_cuda::mem::initModule(env, mem);
  node_cuda::program::initModule(env, program);
  node_cuda::stream::initModule(env, stream);

  EXPORT_ENUM(env, exports, "VERSION", CUDA_VERSION);
  EXPORT_ENUM(env, exports, "IPC_HANDLE_SIZE", CU_IPC_HANDLE_SIZE);

  return exports;
}

NODE_API_MODULE(node_cuda, initModule);
