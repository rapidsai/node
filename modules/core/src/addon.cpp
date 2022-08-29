// Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <nv_node/addon.hpp>
#include <nv_node/utilities/args.hpp>
#include <nv_node/utilities/napi_to_cpp.hpp>

#include <napi.h>

#include <nvml.h>

struct rapidsai_core : public nv::EnvLocalAddon, public Napi::Addon<rapidsai_core> {
  rapidsai_core(Napi::Env const& env, Napi::Object exports) : EnvLocalAddon(env, exports) {
    _after_init = Napi::Persistent(Napi::Function::New(env, [](Napi::CallbackInfo const& info) {
      auto env = info.Env();
      if (nvmlInit_v2() != NVML_SUCCESS) {
        throw Napi::Error::New(env, "Failed to initialize nvml.");
      }
    }));
    DefineAddon(
      exports,
      {
        InstanceValue("_cpp_exports", _cpp_exports.Value()),
        InstanceMethod("init", &rapidsai_core::InitAddon),
        InstanceMethod<&rapidsai_core::get_cuda_driver_version>("getCudaDriverVersion"),
        InstanceMethod<&rapidsai_core::get_compute_capabilities>("getComputeCapabilities"),
      });
  }

  Napi::Value get_cuda_driver_version(Napi::CallbackInfo const& info) {
    auto env = info.Env();
    int32_t cuda_version{};
    auto ary = Napi::Array::New(env, 2);
    auto res = nvmlSystemGetCudaDriverVersion(&cuda_version);
    if (res != NVML_SUCCESS) { throw Napi::Error::New(env, nvmlErrorString(res)); }
    if (cuda_version > 0) {
      ary.Set(0u,
              Napi::String::New(env, std::to_string(NVML_CUDA_DRIVER_VERSION_MAJOR(cuda_version))));
      ary.Set(1u,
              Napi::String::New(env, std::to_string(NVML_CUDA_DRIVER_VERSION_MINOR(cuda_version))));
    } else {
      ary.Set(0u, Napi::String::New(env, ""));
      ary.Set(1u, Napi::String::New(env, ""));
    }
    return ary;
  }

  Napi::Value get_compute_capabilities(Napi::CallbackInfo const& info) {
    auto env = info.Env();

    nvmlDevice_t device{};
    uint32_t arch_index{};
    uint32_t device_count{};
    int32_t major{}, minor{};

    auto res = nvmlDeviceGetCount_v2(&device_count);
    if (res != NVML_SUCCESS) { throw Napi::Error::New(env, nvmlErrorString(res)); }

    std::vector<std::string> archs(device_count);

    for (uint32_t device_index{0}; device_index < device_count; ++device_index) {
      res = nvmlDeviceGetHandleByIndex_v2(device_index, &device);
      if (res != NVML_SUCCESS) { throw Napi::Error::New(env, nvmlErrorString(res)); }

      res = nvmlDeviceGetCudaComputeCapability(device, &major, &minor);
      if (res != NVML_SUCCESS) { throw Napi::Error::New(env, nvmlErrorString(res)); }

      archs[arch_index++] = std::to_string(major) + std::to_string(minor);
    }

    auto ary   = Napi::Array::New(env, arch_index);
    arch_index = 0;
    for (auto const& arch : archs) { ary.Set(arch_index++, arch); }
    return ary;

    return Napi::Array::New(env, 0);
  }
};

NODE_API_ADDON(rapidsai_core);
