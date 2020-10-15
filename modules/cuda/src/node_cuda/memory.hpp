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

#include "node_cuda/utilities/cpp_to_napi.hpp"

#include <nv_node/utilities/args.hpp>

#include <cuda_runtime_api.h>
#include <napi.h>
#include <cstdint>
#include <tuple>

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
  Memory(Napi::CallbackInfo const& args) {}

  void* data() const { return data_; }
  size_t size() const { return size_; }
  int32_t device() const { return device_id_; }
  uint8_t* base() const { return reinterpret_cast<uint8_t*>(data_); }
  uintptr_t ptr() const { return reinterpret_cast<uintptr_t>(data_); }

 protected:
  Napi::Value device(Napi::CallbackInfo const& info) { return CPPToNapi(info)(device()); }
  Napi::Value ptr(Napi::CallbackInfo const& info) { return CPPToNapi(info)(ptr()); }
  Napi::Value size(Napi::CallbackInfo const& info) { return CPPToNapi(info)(size_); }

  inline std::pair<int64_t, int64_t> clamp_slice_args(int64_t len, int64_t lhs, int64_t rhs) {
    // Adjust args similar to Array.prototype.slice. Normalize begin/end to
    // clamp between 0 and length, and wrap around on negative indices, e.g.
    // slice(-1, 5) or slice(5, -1)
    //
    // wrap around on negative start/end positions
    if (lhs < 0) { lhs = ((lhs % len) + len) % len; }
    if (rhs < 0) { rhs = ((rhs % len) + len) % len; }
    // enforce lhs <= rhs and rhs <= count
    return rhs < lhs ? std::make_pair(rhs, lhs) : std::make_pair(lhs, rhs > len ? len : rhs);
  }

  void* data_{nullptr};  ///< Pointer to memory allocation
  size_t size_{0};       ///< Requested size of the memory allocation
  int32_t device_id_{0};
};

/**
 * @brief An owning wrapper around a pinned host memory allocation.
 *
 */
class PinnedMemory : public Napi::ObjectWrap<PinnedMemory>, public Memory {
 public:
  /**
   * @brief Initialize and export the PinnedMemory JavaScript constructor and prototype.
   *
   * @param env The active JavaScript environment.
   * @param exports The exports object to decorate.
   * @return Napi::Object The decorated exports object.
   */
  static Napi::Object Init(Napi::Env env, Napi::Object exports);

  /**
   * @brief Construct a new PinnedMemory instance from C++.
   *
   * @param size Size in bytes to allocate in pinned host memory.
   */
  static Napi::Object New(size_t size);

  /**
   * @brief Check whether an Napi value is an instance of `PinnedMemory`.
   *
   * @param val The Napi::Value to test
   * @return true if the value is a `PinnedMemory`
   * @return false if the value is not a `PinnedMemory`
   */
  inline static bool is_instance(Napi::Value const& val) {
    return val.IsObject() and val.As<Napi::Object>().InstanceOf(constructor.Value());
  }

  /**
   * @brief Construct a new PinnedMemory instance from JavaScript.
   *
   * @param args The JavaScript arguments list wrapped in a conversion helper.
   */
  PinnedMemory(CallbackArgs const& args);

  /**
   * @brief Initialize the PinnedMemory instance created by either C++ or JavaScript.
   *
   * @param size Size in bytes to allocate in pinned host memory.
   */
  void Initialize(size_t size);

  /**
   * @brief Destructor called when the JavaScript VM garbage collects this PinnedMemory instance.
   *
   * @param env The active JavaScript environment.
   */
  void Finalize(Napi::Env env) override;

 private:
  static Napi::FunctionReference constructor;

  Napi::Value slice(Napi::CallbackInfo const& info);
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
   * @brief Check whether an Napi value is an instance of `DeviceMemory`.
   *
   * @param val The Napi::Value to test
   * @return true if the value is a `DeviceMemory`
   * @return false if the value is not a `DeviceMemory`
   */
  inline static bool is_instance(Napi::Value const& val) {
    return val.IsObject() and val.As<Napi::Object>().InstanceOf(constructor.Value());
  }

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

  Napi::Value slice(Napi::CallbackInfo const& info);
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
   * @brief Check whether an Napi value is an instance of `ManagedMemory`.
   *
   * @param val The Napi::Value to test
   * @return true if the value is a `ManagedMemory`
   * @return false if the value is not a `ManagedMemory`
   */
  inline static bool is_instance(Napi::Value const& val) {
    return val.IsObject() and val.As<Napi::Object>().InstanceOf(constructor.Value());
  }

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

  Napi::Value slice(Napi::CallbackInfo const& info);
};

/**
 * @brief An owning wrapper around a CUDA managed memory allocation.
 *
 */
class IpcMemory : public Napi::ObjectWrap<IpcMemory>, public Memory {
 public:
  /**
   * @brief Initialize and export the IPCMemory JavaScript constructor and prototype.
   *
   * @param env The active JavaScript environment.
   * @param exports The exports object to decorate.
   * @return Napi::Object The decorated exports object.
   */
  static Napi::Object Init(Napi::Env env, Napi::Object exports);

  /**
   * @brief Construct a new IPCMemory instance from C++.
   *
   * @param size Size in bytes to allocate in CUDA managed memory.
   */
  static Napi::Object New(cudaIpcMemHandle_t const& handle);

  /**
   * @brief Check whether an Napi value is an instance of `IpcMemory`.
   *
   * @param val The Napi::Value to test
   * @return true if the value is a `IpcMemory`
   * @return false if the value is not a `IpcMemory`
   */
  inline static bool is_instance(Napi::Value const& val) {
    return val.IsObject() and val.As<Napi::Object>().InstanceOf(constructor.Value());
  }

  /**
   * @brief Construct a new IPCMemory instance from JavaScript.
   *
   * @param args The JavaScript arguments list wrapped in a conversion helper.
   */
  IpcMemory(CallbackArgs const& args);

  /**
   * @brief Initialize the IPCMemory instance created by either C++ or JavaScript.
   *
   * @param size Size in bytes to allocate in CUDA managed memory.
   */
  void Initialize(cudaIpcMemHandle_t const& handle);

  /**
   * @brief Destructor called when the JavaScript VM garbage collects this IPCMemory instance.
   *
   * @param env The active JavaScript environment.
   */
  void Finalize(Napi::Env env) override;

  /**
   * @brief Close the underlying IPC memory handle, allowing the exporting process to free the
   * underlying device memory.
   *
   */
  void close();
  void close(Napi::Env const& env);

 private:
  static Napi::FunctionReference constructor;

  Napi::Value slice(Napi::CallbackInfo const& info);
  Napi::Value close(Napi::CallbackInfo const& info);
};

class IpcHandle : public Napi::ObjectWrap<IpcHandle> {
 public:
  /**
   * @brief Initialize and export the IpcMemoryHandle JavaScript constructor and prototype.
   *
   * @param env The active JavaScript environment.
   * @param exports The exports object to decorate.
   * @return Napi::Object The decorated exports object.
   */
  static Napi::Object Init(Napi::Env env, Napi::Object exports);

  /**
   * @brief Construct a new IpcMemoryHandle instance from C++.
   *
   * @param dmem Device memory for which to create an IPC memory handle.
   */
  static Napi::Object New(DeviceMemory const& dmem);

  /**
   * @brief Check whether an Napi value is an instance of `IpcHandle`.
   *
   * @param val The Napi::Value to test
   * @return true if the value is a `IpcHandle`
   * @return false if the value is not a `IpcHandle`
   */
  inline static bool is_instance(Napi::Value const& val) {
    return val.IsObject() and val.As<Napi::Object>().InstanceOf(constructor.Value());
  }

  /**
   * @brief Construct a new IpcMemoryHandle instance from JavaScript.
   *
   * @param args The JavaScript arguments list wrapped in a conversion helper.
   */
  IpcHandle(CallbackArgs const& args);

  /**
   * @brief Initialize the IpcMemoryHandle instance created by either C++ or JavaScript.
   *
   * @param dmem Device memory for which to create an IPC memory handle.
   */
  void Initialize(DeviceMemory const& dmem);

  /**
   * @brief Destructor called when the JavaScript VM garbage collects this IpcMemoryHandle instance.
   *
   * @param env The active JavaScript environment.
   */
  void Finalize(Napi::Env env) override;

  int32_t device() const {
    if (!dmem_.IsEmpty()) {  //
      return DeviceMemory::Unwrap(dmem_.Value())->device();
    }
    return -1;
  }

  cudaIpcMemHandle_t* handle() const {
    return reinterpret_cast<cudaIpcMemHandle_t*>(handle_.Value().Data());
  }

  /**
   * @brief Close the underlying IPC memory handle, allowing this process to free the underlying
   * device memory.
   */
  void close();
  void close(Napi::Env const& env);

 private:
  static Napi::FunctionReference constructor;

  Napi::ObjectReference dmem_;
  Napi::Reference<Napi::Uint8Array> handle_;

  Napi::Value buffer(Napi::CallbackInfo const& info);
  Napi::Value device(Napi::CallbackInfo const& info);
  Napi::Value handle(Napi::CallbackInfo const& info);
  Napi::Value close(Napi::CallbackInfo const& info);
};

/**
 * @brief An owning wrapper around a CUDA managed memory allocation.
 *
 */
class MappedGLMemory : public Napi::ObjectWrap<MappedGLMemory>, public Memory {
 public:
  /**
   * @brief Initialize and export the MappedGLMemory JavaScript constructor and prototype.
   *
   * @param env The active JavaScript environment.
   * @param exports The exports object to decorate.
   * @return Napi::Object The decorated exports object.
   */
  static Napi::Object Init(Napi::Env env, Napi::Object exports);

  /**
   * @brief Construct a new MappedGLMemory instance from C++.
   *
   * @param resource The registered CUDA Graphics Resource for an OpenGL buffer.
   */
  static Napi::Object New(cudaGraphicsResource_t resource);

  /**
   * @brief Check whether an Napi value is an instance of `MappedGLMemory`.
   *
   * @param val The Napi::Value to test
   * @return true if the value is a `MappedGLMemory`
   * @return false if the value is not a `MappedGLMemory`
   */
  inline static bool is_instance(Napi::Value const& val) {
    return val.IsObject() and val.As<Napi::Object>().InstanceOf(constructor.Value());
  }

  /**
   * @brief Construct a new MappedGLMemory instance from JavaScript.
   *
   * @param args The JavaScript arguments list wrapped in a conversion helper.
   */
  MappedGLMemory(CallbackArgs const& args);

  /**
   * @brief Initialize the MappedGLMemory instance created by either C++ or JavaScript.
   *
   * @param resource The registered CUDA Graphics Resource for an OpenGL buffer.
   */
  void Initialize(cudaGraphicsResource_t resource);

  /**
   * @brief Destructor called when the JavaScript VM garbage collects this MappedGLMemory instance.
   *
   * @param env The active JavaScript environment.
   */
  void Finalize(Napi::Env env) override;

 private:
  static Napi::FunctionReference constructor;

  Napi::Value slice(Napi::CallbackInfo const& info);
};

}  // namespace nv
