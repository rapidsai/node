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

#include "utilities/cpp_to_napi.hpp"

#include <node_cuda/device.hpp>

#include <nv_node/utilities/wrap.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/binning_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/fixed_size_memory_resource.hpp>
#include <rmm/mr/device/logging_resource_adaptor.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/thread_safe_resource_adaptor.hpp>

#include <napi.h>
#include <memory>

namespace nv {

struct MemoryResource : public Napi::ObjectWrap<MemoryResource> {
  static Napi::Object Init(Napi::Env env, Napi::Object exports);

  inline static ObjectUnwrap<MemoryResource> Cuda() { return constructor.New(mr_type::cuda); }

  inline static ObjectUnwrap<MemoryResource> Cuda(int32_t device_id) {
    return constructor.New(mr_type::cuda, device_id);
  }

  inline static ObjectUnwrap<MemoryResource> Managed(
    int32_t device_id = Device::active_device_id()) {
    return constructor.New(mr_type::managed, device_id);
  }

  inline static ObjectUnwrap<MemoryResource> Pool(Napi::Object const& upstream_mr,
                                                  size_t initial_pool_size = -1,
                                                  size_t maximum_pool_size = -1) {
    return constructor.New(mr_type::pool, upstream_mr, initial_pool_size, maximum_pool_size);
  }

  inline static ObjectUnwrap<MemoryResource> FixedSize(Napi::Object const& upstream_mr,
                                                       size_t block_size            = 1 << 20,
                                                       size_t blocks_to_preallocate = 128) {
    return constructor.New(mr_type::fixedsize, upstream_mr, block_size, blocks_to_preallocate);
  }

  inline static ObjectUnwrap<MemoryResource> Binning(Napi::Object const& upstream_mr,
                                                     size_t min_size_exponent = -1,
                                                     size_t max_size_exponent = -1) {
    return constructor.New(mr_type::binning, upstream_mr, min_size_exponent, max_size_exponent);
  }

  inline static ObjectUnwrap<MemoryResource> Logging(Napi::Object const& upstream_mr,
                                                     std::string const& log_file_path = "",
                                                     bool auto_flush                  = false) {
    return constructor.New(mr_type::logging, upstream_mr, log_file_path, auto_flush);
  }

  /**
   * @brief Check whether an Napi object is an instance of `MemoryResource`.
   *
   * @param val The Napi::Object to test
   * @return true if the object is a `MemoryResource`
   * @return false if the object is not a `MemoryResource`
   */
  inline static bool is_instance(Napi::Object const& val) {
    return val.InstanceOf(constructor.Value());
  }
  /**
   * @brief Check whether an Napi value is an instance of `MemoryResource`.
   *
   * @param val The Napi::Value to test
   * @return true if the value is a `MemoryResource`
   * @return false if the value is not a `MemoryResource`
   */
  inline static bool is_instance(Napi::Value const& val) {
    return val.IsObject() and is_instance(val.As<Napi::Object>());
  }

  MemoryResource(CallbackArgs const& args);

  inline operator rmm::mr::device_memory_resource*() const { return mr_.get(); }

  /**
   * @brief Get the device id for the MemoryResource.
   *
   * @return ValueWrap<int32_t> The wrapped Device id
   */
  ValueWrap<int32_t> device() const;

  ValueWrap<std::string const> file_path() const;

  /**
   * @copydoc rmm::mr::device_memory_resource::is_equal(
   *            rmm::mr::device_memory_resource const& other)
   */
  ValueWrap<bool> is_equal(rmm::mr::device_memory_resource const& other) const noexcept;

  /**
   * @copydoc rmm::mr::device_memory_resource::get_mem_info(
   *            rmm::cuda_stream_view stream)
   */
  ValueWrap<std::pair<std::size_t, std::size_t>> get_mem_info(rmm::cuda_stream_view stream) const;

  /**
   * @copydoc rmm::mr::device_memory_resource::supports_streams()
   */
  ValueWrap<bool> supports_streams() const noexcept;

  /**
   * @copydoc rmm::mr::device_memory_resource::supports_get_mem_info()
   */
  ValueWrap<bool> supports_get_mem_info() const noexcept;

  /**
   * @copydoc rmm::mr::logging_resource_adaptor::flush()
   */
  void flush();

  /**
   * @brief Adds a bin of the specified maximum allocation size to this memory resource. If
   * specified, uses bin_resource for allocation for this bin. If not specified, creates and uses a
   * FixedSizeMemoryResource for allocation for this bin.
   *
   * Allocations smaller than allocation_size and larger than the next smaller bin size will use
   * this fixed-size memory resource.
   *
   * @param allocation_size The maximum allocation size in bytes for the created bin
   * @param bin_resource The resource to use for this bin (optional)
   * @return void
   */
  void add_bin(size_t allocation_size);
  void add_bin(size_t allocation_size, ObjectUnwrap<MemoryResource> const& bin_resource);

 private:
  static ConstructorReference constructor;

  inline rmm::mr::binning_memory_resource<rmm::mr::device_memory_resource>* get_bin_mr() {
    return static_cast<rmm::mr::binning_memory_resource<rmm::mr::device_memory_resource>*>(
      mr_.get());
  }

  inline rmm::mr::logging_resource_adaptor<rmm::mr::device_memory_resource>* get_log_mr() {
    return static_cast<rmm::mr::logging_resource_adaptor<rmm::mr::device_memory_resource>*>(
      mr_.get());
  }

  Napi::Value flush(Napi::CallbackInfo const& info);
  Napi::Value add_bin(Napi::CallbackInfo const& info);
  Napi::Value is_equal(Napi::CallbackInfo const& info);

  Napi::Value get_device(Napi::CallbackInfo const& info);
  Napi::Value get_mem_info(Napi::CallbackInfo const& info);
  Napi::Value get_file_path(Napi::CallbackInfo const& info);
  Napi::Value get_upstream_mr(Napi::CallbackInfo const& info);

  Napi::Value supports_streams(Napi::CallbackInfo const& info);
  Napi::Value supports_get_mem_info(Napi::CallbackInfo const& info);

  int32_t device_id_{-1};
  std::string log_file_path_{};
  mr_type type_{mr_type::cuda};

  Napi::ObjectReference upstream_mr_;
  std::vector<Napi::ObjectReference> bin_mrs_;
  std::shared_ptr<rmm::mr::device_memory_resource> mr_;
};

}  // namespace nv
