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

#include "rmm/device_buffer.hpp"

#include <nv_node/utilities/span.hpp>

#include <napi.h>
#include <memory>

namespace nv {

class DeviceBuffer : public Napi::ObjectWrap<DeviceBuffer> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);

  /**
   * @brief Construct a new DeviceBuffer instance from C++.
   *
   * @param data Pointer to the host or device memory to copy from.
   * @param stream CUDA stream on which memory may be allocated if the memory
   * resource supports streams.
   * @param mr Memory resource to use for the device memory allocation.
   */
  static Napi::Object New(
    Span<char> span,
    cudaStream_t stream                 = 0,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) {
    return DeviceBuffer::New(span.data(), span.size(), stream, mr);
  }

  /**
   * @brief Construct a new DeviceBuffer instance from C++.
   *
   * @param data Pointer to the host or device memory to copy from.
   * @param size Size in bytes to copy.
   * @param stream CUDA stream on which memory may be allocated if the memory
   * resource supports streams.
   * @param mr Memory resource to use for the device memory allocation.
   */
  static Napi::Object New(
    void* data,
    size_t size,
    cudaStream_t stream                 = 0,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Construct a DeviceBuffer by copying an existing rmm::device_buffer.
   *
   * @param other The `device_buffer` whose contents will be copied into the
   * newly constructed one.
   */
  static Napi::Object New(rmm::device_buffer const& other);

  /**
   * @brief Construct a DeviceBuffer by moving an existing rmm::device_buffer.
   *
   * @param other The `device_buffer` whose contents will be moved into the
   * newly constructed one.
   */
  static Napi::Object New(rmm::device_buffer&& other);

  /**
   * @brief Check whether an Napi value is an instance of `DeviceBuffer`.
   *
   * @param val The Napi::Value to test
   * @return true if the value is a `DeviceBuffer`
   * @return false if the value is not a `DeviceBuffer`
   */
  inline static bool is_instance(Napi::Value const& val) {
    return val.IsObject() and val.As<Napi::Object>().InstanceOf(constructor.Value());
  }

  /**
   * @brief Construct a new DeviceBuffer instance from JavaScript.
   *
   */
  DeviceBuffer(Napi::CallbackInfo const& info);

  /**
   * @brief Initialize the DeviceBuffer instance created by either C++ or JavaScript.
   *
   * @param data Pointer to the host or device memory to copy from.
   * @param size Size in bytes to copy.
   * @param stream CUDA stream on which memory may be allocated if the memory
   * resource supports streams.
   * @param mr Memory resource to use for the device memory allocation.
   */
  void Initialize(void* data,
                  size_t size,
                  cudaStream_t stream                 = 0,
                  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Destructor called when the JavaScript VM garbage collects this DeviceBuffer
   * instance.
   *
   * @param env The active JavaScript environment.
   */
  void Finalize(Napi::Env env) override;

  void* data() const { return Buffer()->data(); }
  size_t size() const { return Buffer()->size(); }

 private:
  static Napi::FunctionReference constructor;

  char* Data() const { return static_cast<char*>(Buffer()->data()); }
  std::unique_ptr<rmm::device_buffer> const& Buffer() const { return buffer_; }

  Napi::Value byteLength(Napi::CallbackInfo const& info);
  Napi::Value capacity(Napi::CallbackInfo const& info);
  Napi::Value isEmpty(Napi::CallbackInfo const& info);
  Napi::Value ptr(Napi::CallbackInfo const& info);
  Napi::Value stream(Napi::CallbackInfo const& info);
  Napi::Value resize(Napi::CallbackInfo const& info);
  Napi::Value setStream(Napi::CallbackInfo const& info);
  Napi::Value shrinkToFit(Napi::CallbackInfo const& info);
  Napi::Value slice(Napi::CallbackInfo const& info);

  std::unique_ptr<rmm::device_buffer> buffer_;
};

}  // namespace nv
