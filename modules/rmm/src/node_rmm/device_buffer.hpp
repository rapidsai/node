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
#include <nv_node/utilities/wrap.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <napi.h>
#include <memory>

namespace nv {

struct DeviceBuffer : public Napi::ObjectWrap<DeviceBuffer> {
  static Napi::Object Init(Napi::Env env, Napi::Object exports);

  /**
   * @brief Construct a new DeviceBuffer instance from an rmm::device_buffer.
   *
   * @param buffer Pointer the rmm::device_buffer to own.
   */
  static ObjectUnwrap<DeviceBuffer> New(std::unique_ptr<rmm::device_buffer> buffer);

  /**
   * @brief Construct a new uninitialized DeviceBuffer instance from C++.
   *
   * @param mr Memory resource to use for the device memory allocation.
   * @param stream CUDA stream on which memory may be allocated if the memory
   * resource supports streams.
   */
  inline static ObjectUnwrap<DeviceBuffer> New(
    ObjectUnwrap<MemoryResource> const& mr = MemoryResource::Cuda(),
    rmm::cuda_stream_view stream           = rmm::cuda_stream_default) {
    return DeviceBuffer::New(nullptr, 0, mr, stream);
  }

  /**
   * @brief Construct a new DeviceBuffer instance from C++.
   *
   * @param data Pointer to the host or device memory to copy from.
   * @param stream CUDA stream on which memory may be allocated if the memory
   * resource supports streams.
   */
  inline static ObjectUnwrap<DeviceBuffer> New(
    Span<char> const& data,
    ObjectUnwrap<MemoryResource> const& mr = MemoryResource::Cuda(),
    rmm::cuda_stream_view stream           = rmm::cuda_stream_default) {
    return DeviceBuffer::New(data.data(), data.size(), mr, stream);
  }

  /**
   * @brief Construct a new DeviceBuffer instance from an Array.
   *
   * @param data Array to copy from.
   * @param stream CUDA stream on which memory may be allocated if the memory
   * resource supports streams.
   */
  template <typename T = double>
  inline static ObjectUnwrap<DeviceBuffer> New(
    Napi::Array const& data,
    ObjectUnwrap<MemoryResource> const& mr = MemoryResource::Cuda(),
    rmm::cuda_stream_view stream           = rmm::cuda_stream_default) {
    std::vector<T> hvec = NapiToCPP{data};
    return New(hvec.data(), hvec.size() * sizeof(T), mr.object(), stream);
  }

  /**
   * @brief Construct a new DeviceBuffer instance from an ArrayBuffer.
   *
   * @param data ArrayBuffer to host memory to copy from.
   * @param stream CUDA stream on which memory may be allocated if the memory
   * resource supports streams.
   */
  static ObjectUnwrap<DeviceBuffer> New(
    Napi::ArrayBuffer const& data,
    ObjectUnwrap<MemoryResource> const& mr = MemoryResource::Cuda(),
    rmm::cuda_stream_view stream           = rmm::cuda_stream_default);

  /**
   * @brief Construct a new DeviceBuffer instance from C++.
   *
   * @param data Pointer to the host or device memory to copy from.
   * @param size Size in bytes to copy.
   * @param stream CUDA stream on which memory may be allocated if the memory
   * resource supports streams.
   * @param mr Memory resource to use for the device memory allocation.
   */
  static ObjectUnwrap<DeviceBuffer> New(
    void* const data,
    size_t const size,
    ObjectUnwrap<MemoryResource> const& mr = MemoryResource::Cuda(),
    rmm::cuda_stream_view stream           = rmm::cuda_stream_default);

  /**
   * @brief Check whether an Napi object is an instance of `DeviceBuffer`.
   *
   * @param val The Napi::Object to test
   * @return true if the object is a `DeviceBuffer`
   * @return false if the object is not a `DeviceBuffer`
   */
  inline static bool is_instance(Napi::Object const& val) {
    return val.InstanceOf(constructor.Value());
  }
  /**
   * @brief Check whether an Napi value is an instance of `DeviceBuffer`.
   *
   * @param val The Napi::Value to test
   * @return true if the value is a `DeviceBuffer`
   * @return false if the value is not a `DeviceBuffer`
   */
  inline static bool is_instance(Napi::Value const& val) {
    return val.IsObject() and is_instance(val.As<Napi::Object>());
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

  inline void* data() const { return buffer_->data(); }

  inline size_t size() const { return buffer_->size(); }

  inline bool is_empty() const { return buffer_->is_empty(); }

  inline ValueWrap<rmm::cuda_stream_view> stream() { return {Env(), buffer_->stream()}; }

  ValueWrap<int32_t> device() const;

  inline explicit operator rmm::cuda_device_id() const { return NapiToCPP(mr_.Value()); }

  inline rmm::mr::device_memory_resource* get_mr() const { return NapiToCPP(mr_.Value()); }

  inline operator Napi::Value() const { return Value(); }

 private:
  static ConstructorReference constructor;

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

  std::unique_ptr<rmm::device_buffer> buffer_;  ///< Pointer to the underlying rmm::device_buffer
  Napi::ObjectReference mr_;  ///< Reference to the JS MemoryResource used by this device_buffer
};

}  // namespace nv
