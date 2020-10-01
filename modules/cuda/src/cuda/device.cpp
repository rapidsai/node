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
                });
  Device::constructor = Napi::Persistent(ctor);
  Device::constructor.SuppressDestruct();

  exports.Set("Device", ctor);

  return exports;
}

Device::Device(CallbackArgs const& args) : Napi::ObjectWrap<Device>(args) {
  NODE_CUDA_EXPECT(args.IsConstructCall(), "Device constructor requires 'new'");
  NODE_CUDA_EXPECT(args.Length() == 0 || (args.Length() == 1 && args[0].IsNumber()),
                   "Device constructor requires a numeric deviceId argument");
  switch (args.Length()) {
    case 0: Initialize(); break;
    case 1: Initialize(args[0]); break;
  }
}

Napi::Object Device::New(int32_t id) {
  auto inst = Device::constructor.New({});
  Device::Unwrap(inst)->Initialize(id);
  return inst;
}

void Device::Initialize(int32_t id) {
  id_ = id;
  NODE_CUDA_TRY(cudaGetDeviceProperties(&props_, id));
  auto bus_id = const_cast<char*>(pci_bus_name_.data());
  NODE_CUDA_TRY(cudaDeviceGetPCIBusId(bus_id, pci_bus_name_.size(), id));
  pci_bus_name_.shrink_to_fit();
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

}  // namespace nv
