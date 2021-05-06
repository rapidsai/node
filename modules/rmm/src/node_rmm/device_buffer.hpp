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

#pragma once

#include <node_rmm/memory_resource.hpp>
#include <node_rmm/utilities/napi_to_cpp.hpp>

#include <nv_node/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <napi.h>
#include <memory>

namespace nv {

struct DeviceBuffer : public EnvLocalObjectWrap<DeviceBuffer> {
  /**
   * @brief Initialize and export the DeviceBuffer JavaScript constructor and prototype.
   *
   * @param env The active JavaScript environment.
   * @param exports The exports object to decorate.
   * @return Napi::Function The DeviceBuffer constructor function.
   */
  static Napi::Function Init(Napi::Env const& env, Napi::Object exports);

  /**
   * @brief Construct a new DeviceBuffer instance from an rmm::device_buffer.
   *
   * @param buffer Pointer the rmm::device_buffer to own.
   */
  static wrapper_t New(Napi::Env const& env, std::unique_ptr<rmm::device_buffer> buffer);

  /**
   * @brief Construct a new uninitialized DeviceBuffer instance from C++.
   *
   * @param mr Memory resource to use for the device memory allocation.
   * @param stream CUDA stream on which memory may be allocated if the memory
   * resource supports streams.
   */
  inline static wrapper_t New(Napi::Env const& env,
                              MemoryResource::wrapper_t const& mr,
                              rmm::cuda_stream_view stream = rmm::cuda_stream_default) {
    return EnvLocalObjectWrap<DeviceBuffer>::New(env, 0, mr, stream);
  }

  /**
   * @brief Construct a new uninitialized DeviceBuffer instance from C++.
   *
   * @param stream CUDA stream on which memory may be allocated if the memory
   * resource supports streams.
   */
  inline static wrapper_t New(Napi::Env const& env,
                              rmm::cuda_stream_view stream = rmm::cuda_stream_default) {
    return EnvLocalObjectWrap<DeviceBuffer>::New(env, 0, MemoryResource::Cuda(env), stream);
  }

  /**
   * @brief Construct a new DeviceBuffer instance from an Array.
   *
   * @param data Array to copy from.
   * @param mr Memory resource to use for the device memory allocation.
   * @param stream CUDA stream on which memory may be allocated if the memory
   * resource supports streams.
   */
  template <typename T = double>
  inline static wrapper_t New(Napi::Env const& env,
                              Napi::Array const& data,
                              MemoryResource::wrapper_t const& mr,
                              rmm::cuda_stream_view stream = rmm::cuda_stream_default) {
    std::vector<T> hvec = NapiToCPP{data};
    return New(env, Span<char>{hvec.data(), hvec.size()}, mr, stream);
  }

  /**
   * @brief Construct a new DeviceBuffer instance from a Uint8Array.
   *
   * @param data Uint8Array of host memory to copy from.
   * @param stream CUDA stream on which memory may be allocated if the memory
   * resource supports streams.
   */
  static wrapper_t New(Napi::Env const& env,
                       Napi::Uint8Array const& data,
                       MemoryResource::wrapper_t const& mr,
                       rmm::cuda_stream_view stream = rmm::cuda_stream_default);

  /**
   * @brief Construct a new DeviceBuffer instance from C++.
   *
   * @param data Pointer to the host or device memory to copy from.
   * @param stream CUDA stream on which memory may be allocated if the memory
   * resource supports streams.
   */
  static wrapper_t New(Napi::Env const& env,
                       Span<char> const& data,
                       MemoryResource::wrapper_t const& mr,
                       rmm::cuda_stream_view stream = rmm::cuda_stream_default);

  /**
   * @brief Construct a new DeviceBuffer instance from C++.
   *
   * @param data Pointer to the host or device memory to copy from.
   * @param size Size in bytes to copy.
   * @param stream CUDA stream on which memory may be allocated if the memory
   * resource supports streams.
   * @param mr Memory resource to use for the device memory allocation.
   */
  static wrapper_t New(Napi::Env const& env,
                       void* const data,
                       size_t const size,
                       MemoryResource::wrapper_t const& mr,
                       rmm::cuda_stream_view stream = rmm::cuda_stream_default);

  /**
   * @brief Construct a new DeviceBuffer instance.
   *
   */
  DeviceBuffer(CallbackArgs const& info);

  /**
   * @brief Destructor called when the JavaScript VM garbage collects this DeviceBuffer
   * instance.
   *
   * @param env The active JavaScript environment.
   */
  void Finalize(Napi::Env env) override;

  void dispose(Napi::Env env);

  rmm::cuda_device_id device() const noexcept;

  inline void* data() const { return buffer_.get() != nullptr ? buffer_->data() : nullptr; }

  inline size_t size() const { return buffer_.get() != nullptr ? buffer_->size() : 0; }

  inline size_t capacity() const { return buffer_.get() != nullptr ? buffer_->capacity() : 0; }

  inline bool is_empty() const { return buffer_.get() != nullptr ? buffer_->is_empty() : true; }

  inline rmm::cuda_stream_view stream() {
    return buffer_.get() != nullptr ? buffer_->stream() : rmm::cuda_stream_default;
  }

  inline rmm::mr::device_memory_resource* get_mr() const {
    return mr_.Value()->operator rmm::mr::device_memory_resource*();
  }

  // convert to void*
  inline operator void*() const { return static_cast<void*>(data()); }
  // convert to const void*
  inline operator const void*() const { return static_cast<void*>(data()); }

  // convert to cudf::valid_type*
  inline operator uint8_t*() const { return static_cast<uint8_t*>(data()); }
  // convert to const cudf::valid_type*
  inline operator const uint8_t*() const { return static_cast<uint8_t*>(data()); }

  // convert to cudf::offset_type*
  inline operator int32_t*() const { return static_cast<int32_t*>(data()); }
  // convert to const cudf::offset_type*
  inline operator const int32_t*() const { return static_cast<int32_t*>(data()); }

  // convert to cudf::bitmask_type*
  inline operator uint32_t*() const { return static_cast<uint32_t*>(data()); }
  // convert to const cudf::bitmask_type*
  inline operator const uint32_t*() const { return static_cast<uint32_t*>(data()); }

 private:
  Napi::Value get_mr(Napi::CallbackInfo const& info);
  Napi::Value byte_length(Napi::CallbackInfo const& info);
  Napi::Value capacity(Napi::CallbackInfo const& info);
  Napi::Value is_empty(Napi::CallbackInfo const& info);
  Napi::Value ptr(Napi::CallbackInfo const& info);
  Napi::Value device(Napi::CallbackInfo const& info);
  Napi::Value stream(Napi::CallbackInfo const& info);
  void resize(Napi::CallbackInfo const& info);
  void set_stream(Napi::CallbackInfo const& info);
  void shrink_to_fit(Napi::CallbackInfo const& info);
  Napi::Value slice(Napi::CallbackInfo const& info);

  void dispose(Napi::CallbackInfo const&);

  std::unique_ptr<rmm::device_buffer> buffer_;  ///< Pointer to the underlying rmm::device_buffer
  Napi::Reference<MemoryResource::wrapper_t>
    mr_;  ///< Reference to the JS MemoryResource used by this device_buffer
  rmm::cuda_device_id device_id_{Device::active_device_id()};
};

}  // namespace nv
