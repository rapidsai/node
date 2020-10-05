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

#pragma once

#include "node_cuda/device.hpp"
#include "node_cuda/utilities/cpp_to_napi.hpp"
#include "node_cuda/utilities/error.hpp"

#include <napi.h>

#include <nv_node/utilities/args.hpp>

#include <cstdint>

namespace nv {

/**
 * @brief Base class for an owning wrapper around a memory allocation.
 *
 */
class Memory {
 public:
  /**
   * @brief Construct a new Memory instance from JavaScript.
   *
   * @param args The JavaScript arguments list wrapped in a conversion helper.
   */
  Memory(CallbackArgs const& args) {
    NODE_CUDA_EXPECT(args.IsConstructCall(), "Memory constructor requires 'new'");
    NODE_CUDA_EXPECT(args.Length() == 0 || (args.Length() == 1 && args[0].IsNumber()),
                     "Memory constructor requires a numeric byteLength argument");
  }

  void* data() { return data_; }
  size_t size() { return size_; }
  int32_t device() { return device_id_; }
  uint8_t* base() { return reinterpret_cast<uint8_t*>(data_); }
  uintptr_t ptr() { return reinterpret_cast<uintptr_t>(data_); }

 protected:
  Napi::Value GetDevice(Napi::CallbackInfo const& info) { return CPPToNapi(info)(device()); }
  Napi::Value GetPointer(Napi::CallbackInfo const& info) { return CPPToNapi(info)(ptr()); }
  Napi::Value GetByteLength(Napi::CallbackInfo const& info) { return CPPToNapi(info)(size_); }

  void* data_{nullptr};  ///< Pointer to memory allocation
  size_t size_{0};       ///< Requested size of the memory allocation
  int32_t device_id_{0};
};

/**
 * @brief An owning wrapper around a pinned host memory allocation.
 *
 */
class HostMemory : public Napi::ObjectWrap<HostMemory>, public Memory {
 public:
  /**
   * @brief Initialize and export the HostMemory JavaScript constructor and prototype.
   *
   * @param env The active JavaScript environment.
   * @param exports The exports object to decorate.
   * @return Napi::Object The decorated exports object.
   */
  static Napi::Object Init(Napi::Env env, Napi::Object exports);

  /**
   * @brief Construct a new HostMemory instance from C++.
   *
   * @param size Size in bytes to allocate in pinned host memory.
   */
  static Napi::Object New(size_t size);

  /**
   * @brief Construct a new HostMemory instance from JavaScript.
   *
   * @param args The JavaScript arguments list wrapped in a conversion helper.
   */
  HostMemory(CallbackArgs const& args);
  /**
   * @brief Initialize the HostMemory instance created by either C++ or JavaScript.
   *
   * @param size Size in bytes to allocate in pinned host memory.
   */
  void Initialize(size_t size);
  /**
   * @brief Destructor called when the JavaScript VM garbage collects this HostMemory instance.
   *
   * @param env The active JavaScript environment.
   */
  void Finalize(Napi::Env env) override;

 private:
  static Napi::FunctionReference constructor;

  Napi::Value CopySlice(Napi::CallbackInfo const& info);
};

/**
 * @brief An owning wrapper around a device memory allocation.
 *
 */
class DeviceMemory : public Napi::ObjectWrap<DeviceMemory>, public Memory {
 public:
  /**
   * @brief Initialize and export the DeviceMemory JavaScript constructor and prototype.
   *
   * @param env The active JavaScript environment.
   * @param exports The exports object to decorate.
   * @return Napi::Object The decorated exports object.
   */
  static Napi::Object Init(Napi::Env env, Napi::Object exports);

  /**
   * @brief Construct a new DeviceMemory instance from C++.
   *
   * @param size Size in bytes to allocate in device memory.
   */
  static Napi::Object New(size_t size);

  /**
   * @brief Construct a new DeviceMemory instance from JavaScript.
   *
   * @param args The JavaScript arguments list wrapped in a conversion helper.
   */
  DeviceMemory(CallbackArgs const& args);
  /**
   * @brief Initialize the DeviceMemory instance created by either C++ or JavaScript.
   *
   * @param size Size in bytes to allocate in device memory.
   */
  void Initialize(size_t size);
  /**
   * @brief Destructor called when the JavaScript VM garbage collects this DeviceMemory instance.
   *
   * @param env The active JavaScript environment.
   */
  void Finalize(Napi::Env env) override;

 private:
  static Napi::FunctionReference constructor;

  Napi::Value CopySlice(Napi::CallbackInfo const& info);
};

/**
 * @brief An owning wrapper around a CUDA managed memory allocation.
 *
 */
class ManagedMemory : public Napi::ObjectWrap<ManagedMemory>, public Memory {
 public:
  /**
   * @brief Initialize and export the ManagedMemory JavaScript constructor and prototype.
   *
   * @param env The active JavaScript environment.
   * @param exports The exports object to decorate.
   * @return Napi::Object The decorated exports object.
   */
  static Napi::Object Init(Napi::Env env, Napi::Object exports);

  /**
   * @brief Construct a new ManagedMemory instance from C++.
   *
   * @param size Size in bytes to allocate in CUDA managed memory.
   */
  static Napi::Object New(size_t size);

  /**
   * @brief Construct a new ManagedMemory instance from JavaScript.
   *
   * @param args The JavaScript arguments list wrapped in a conversion helper.
   */
  ManagedMemory(CallbackArgs const& args);
  /**
   * @brief Initialize the ManagedMemory instance created by either C++ or JavaScript.
   *
   * @param size Size in bytes to allocate in CUDA managed memory.
   */
  void Initialize(size_t size);
  /**
   * @brief Destructor called when the JavaScript VM garbage collects this ManagedMemory instance.
   *
   * @param env The active JavaScript environment.
   */
  void Finalize(Napi::Env env) override;

 private:
  static Napi::FunctionReference constructor;

  Napi::Value CopySlice(Napi::CallbackInfo const& info);
};

}  // namespace nv
