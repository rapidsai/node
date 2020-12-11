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

#include <node_rmm/utilities/napi_to_cpp.hpp>

#include <nv_node/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <napi.h>
#include <memory>

namespace nv {

class DeviceBuffer : public Napi::ObjectWrap<DeviceBuffer> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);

  /**
   * @brief Construct a new DeviceBuffer instance from an rmm::device_buffer.
   *
   * @param buffer Pointer the rmm::device_buffer to own.
   */
  static DeviceBuffer New(std::unique_ptr<rmm::device_buffer> buffer);

  /**
   * @brief Construct a new uninitialized DeviceBuffer instance from C++.
   *
   * @param data Pointer to the host or device memory to copy from.
   * @param stream CUDA stream on which memory may be allocated if the memory
   * resource supports streams.
   * @param mr Memory resource to use for the device memory allocation.
   */
  static DeviceBuffer New(rmm::cuda_stream_view stream = rmm::cuda_stream_default,
                          Napi::Object const& mr       = CudaMemoryResource::New()) {
    return DeviceBuffer::New(Span<char>(0), stream, mr);
  }

  /**
   * @brief Construct a new DeviceBuffer instance from C++.
   *
   * @param data Pointer to the host or device memory to copy from.
   * @param stream CUDA stream on which memory may be allocated if the memory
   * resource supports streams.
   * @param mr Memory resource to use for the device memory allocation.
   */
  static DeviceBuffer New(Span<char> span,
                          rmm::cuda_stream_view stream = rmm::cuda_stream_default,
                          Napi::Object const& mr       = CudaMemoryResource::New()) {
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
  static DeviceBuffer New(void* data,
                          size_t size,
                          rmm::cuda_stream_view stream = rmm::cuda_stream_default,
                          Napi::Object const& mr       = CudaMemoryResource::New());

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
  DeviceBuffer(CallbackArgs const& info);

  /**
   * @brief Destructor called when the JavaScript VM garbage collects this DeviceBuffer
   * instance.
   *
   * @param env The active JavaScript environment.
   */
  void Finalize(Napi::Env env) override;

  inline void* data() const { return buffer().data(); }

  inline size_t size() const { return buffer().size(); }

  inline rmm::cuda_stream_view stream() { return buffer().stream(); }

  inline int32_t device() const { return (this->operator rmm::cuda_device_id()).value(); }

  inline explicit operator rmm::cuda_device_id() const { return NapiToCPP(mr_.Value()); }

  inline rmm::mr::device_memory_resource* get_mr() const { return NapiToCPP(mr_.Value()); }

 private:
  static Napi::FunctionReference constructor;

  rmm::device_buffer& buffer() const { return *buffer_; }

  Napi::Value get_mr(Napi::CallbackInfo const& info);
  Napi::Value byte_length(Napi::CallbackInfo const& info);
  Napi::Value capacity(Napi::CallbackInfo const& info);
  Napi::Value is_empty(Napi::CallbackInfo const& info);
  Napi::Value ptr(Napi::CallbackInfo const& info);
  Napi::Value device(Napi::CallbackInfo const& info);
  Napi::Value stream(Napi::CallbackInfo const& info);
  Napi::Value resize(Napi::CallbackInfo const& info);
  Napi::Value set_stream(Napi::CallbackInfo const& info);
  Napi::Value shrink_to_fit(Napi::CallbackInfo const& info);
  Napi::Value slice(Napi::CallbackInfo const& info);

  Napi::ObjectReference mr_;  ///< Reference to the JS MemoryResource used by this device_buffer
  std::unique_ptr<rmm::device_buffer> buffer_;  ///< Pointer to the underlying rmm::device_buffer
};

}  // namespace nv
