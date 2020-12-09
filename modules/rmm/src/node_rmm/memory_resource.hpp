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

#include <node_cuda/utilities/napi_to_cpp.hpp>

#include <nv_node/utilities/args.hpp>
#include <nv_node/utilities/cpp_to_napi.hpp>

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

/**
 * @brief Base class for an owning wrapper around an RMM device_memory_resource.
 *
 */
class MemoryResource {
 public:
  /**
   * @brief Check whether an Napi value is an instance of `MemoryResource`.
   *
   * @param val The Napi::Value to test
   * @return true if the value is a `MemoryResource`
   * @return false if the value is not a `MemoryResource`
   */
  static bool is_instance(Napi::Value const& val);

  /**
   * @copydoc rmm::mr::device_memory_resource::is_equal(
   *            rmm::mr::device_memory_resource const& other)
   */
  inline bool is_equal(rmm::mr::device_memory_resource const& other) const noexcept {
    return get_mr()->is_equal(other);
  }

  /**
   * @copydoc rmm::mr::device_memory_resource::get_mem_info(
   *            rmm::cuda_stream_view stream)
   */
  inline std::pair<std::size_t, std::size_t> get_mem_info(rmm::cuda_stream_view stream) {
    return get_mr()->get_mem_info(stream);
  }

  /**
   * @copydoc rmm::mr::device_memory_resource::supports_streams()
   */
  inline bool supports_streams() { return get_mr()->supports_streams(); }

  /**
   * @copydoc rmm::mr::device_memory_resource::supports_get_mem_info()
   */
  inline bool supports_get_mem_info() { return get_mr()->supports_get_mem_info(); }

  inline operator rmm::mr::device_memory_resource*() const { return mr_.get(); }

  std::shared_ptr<rmm::mr::device_memory_resource> get_mr() const { return mr_; }

 protected:
  Napi::Value is_equal(Napi::CallbackInfo const& info);
  Napi::Value get_mem_info(Napi::CallbackInfo const& info);
  Napi::Value supports_streams(Napi::CallbackInfo const& info);
  Napi::Value supports_get_mem_info(Napi::CallbackInfo const& info);

  std::shared_ptr<rmm::mr::device_memory_resource> mr_;
};

class CudaMemoryResource : public Napi::ObjectWrap<CudaMemoryResource>, public MemoryResource {
 public:
  /**
   * @brief Initialize and export the CudaMemoryResource JavaScript constructor and prototype.
   *
   * @param env The active JavaScript environment.
   * @param exports The exports object to decorate.
   * @return Napi::Object The decorated exports object.
   */
  static Napi::Object Init(Napi::Env env, Napi::Object exports);

  /**
   * @brief Construct a new CudaMemoryResource instance from JavaScript.
   *
   */
  CudaMemoryResource(CallbackArgs const& args);

  /**
   * @brief Check whether an Napi value is an instance of `CudaMemoryResource`.
   *
   * @param val The Napi::Value to test
   * @return true if the value is a `CudaMemoryResource`
   * @return false if the value is not a `CudaMemoryResource`
   */
  inline static bool is_instance(Napi::Value const& val) {
    return val.IsObject() and val.As<Napi::Object>().InstanceOf(constructor.Value());
  }

 private:
  static Napi::FunctionReference constructor;
};

class ManagedMemoryResource : public Napi::ObjectWrap<ManagedMemoryResource>,
                              public MemoryResource {
 public:
  /**
   * @brief Initialize and export the ManagedMemoryResource JavaScript constructor and prototype.
   *
   * @param env The active JavaScript environment.
   * @param exports The exports object to decorate.
   * @return Napi::Object The decorated exports object.
   */
  static Napi::Object Init(Napi::Env env, Napi::Object exports);

  /**
   * @brief Construct a new ManagedMemoryResource instance from JavaScript.
   *
   */
  ManagedMemoryResource(CallbackArgs const& args);

  /**
   * @brief Check whether an Napi value is an instance of `ManagedMemoryResource`.
   *
   * @param val The Napi::Value to test
   * @return true if the value is a `ManagedMemoryResource`
   * @return false if the value is not a `ManagedMemoryResource`
   */
  inline static bool is_instance(Napi::Value const& val) {
    return val.IsObject() and val.As<Napi::Object>().InstanceOf(constructor.Value());
  }

 private:
  static Napi::FunctionReference constructor;
};

class PoolMemoryResource : public Napi::ObjectWrap<PoolMemoryResource>, public MemoryResource {
 public:
  /**
   * @brief Initialize and export the PoolMemoryResource JavaScript constructor and prototype.
   *
   * @param env The active JavaScript environment.
   * @param exports The exports object to decorate.
   * @return Napi::Object The decorated exports object.
   */
  static Napi::Object Init(Napi::Env env, Napi::Object exports);

  /**
   * @brief Construct a new PoolMemoryResource instance from JavaScript.
   *
   */
  PoolMemoryResource(CallbackArgs const& args);

  /**
   * @brief Check whether an Napi value is an instance of `PoolMemoryResource`.
   *
   * @param val The Napi::Value to test
   * @return true if the value is a `PoolMemoryResource`
   * @return false if the value is not a `PoolMemoryResource`
   */
  inline static bool is_instance(Napi::Value const& val) {
    return val.IsObject() and val.As<Napi::Object>().InstanceOf(constructor.Value());
  }

 private:
  static Napi::FunctionReference constructor;

  Napi::Value get_upstream_mr(Napi::CallbackInfo const& info);

  Napi::ObjectReference
    upstream_mr_{};  ///< The MemoryResource from which to allocate blocks for the pool.
};

class FixedSizeMemoryResource : public Napi::ObjectWrap<FixedSizeMemoryResource>,
                                public MemoryResource {
 public:
  /**
   * @brief Initialize and export the FixedSizeMemoryResource JavaScript constructor and prototype.
   *
   * @param env The active JavaScript environment.
   * @param exports The exports object to decorate.
   * @return Napi::Object The decorated exports object.
   */
  static Napi::Object Init(Napi::Env env, Napi::Object exports);

  /**
   * @brief Construct a new FixedSizeMemoryResource instance from JavaScript.
   *
   */
  FixedSizeMemoryResource(CallbackArgs const& args);

  /**
   * @brief Check whether an Napi value is an instance of `FixedSizeMemoryResource`.
   *
   * @param val The Napi::Value to test
   * @return true if the value is a `FixedSizeMemoryResource`
   * @return false if the value is not a `FixedSizeMemoryResource`
   */
  inline static bool is_instance(Napi::Value const& val) {
    return val.IsObject() and val.As<Napi::Object>().InstanceOf(constructor.Value());
  }

 private:
  static Napi::FunctionReference constructor;

  Napi::Value get_upstream_mr(Napi::CallbackInfo const& info);

  Napi::ObjectReference
    upstream_mr_{};  ///< The MemoryResource from which to allocate blocks for the pool.
};

class BinningMemoryResource : public Napi::ObjectWrap<BinningMemoryResource>,
                              public MemoryResource {
 public:
  /**
   * @brief Initialize and export the BinningMemoryResource JavaScript constructor and prototype.
   *
   * @param env The active JavaScript environment.
   * @param exports The exports object to decorate.
   * @return Napi::Object The decorated exports object.
   */
  static Napi::Object Init(Napi::Env env, Napi::Object exports);

  /**
   * @brief Construct a new BinningMemoryResource instance from JavaScript.
   *
   */
  BinningMemoryResource(CallbackArgs const& args);

  /**
   * @brief Check whether an Napi value is an instance of `BinningMemoryResource`.
   *
   * @param val The Napi::Value to test
   * @return true if the value is a `BinningMemoryResource`
   * @return false if the value is not a `BinningMemoryResource`
   */
  inline static bool is_instance(Napi::Value const& val) {
    return val.IsObject() and val.As<Napi::Object>().InstanceOf(constructor.Value());
  }

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
  void add_bin(size_t allocation_size, Napi::Object const& bin_resource);

  rmm::mr::binning_memory_resource<rmm::mr::device_memory_resource>* get_bin_mr() {
    return static_cast<rmm::mr::binning_memory_resource<rmm::mr::device_memory_resource>*>(
      mr_.get());
  }

 private:
  static Napi::FunctionReference constructor;

  Napi::Value add_bin(Napi::CallbackInfo const& info);
  Napi::Value get_upstream_mr(Napi::CallbackInfo const& info);

  Napi::ObjectReference
    upstream_mr_{};  ///< The MemoryResource to use for allocations larger than any of the bins.

  std::vector<Napi::ObjectReference> bin_mrs_;
};

class LoggingResourceAdapter : public Napi::ObjectWrap<LoggingResourceAdapter>,
                               public MemoryResource {
 public:
  /**
   * @brief Initialize and export the LoggingResourceAdapter JavaScript constructor and prototype.
   *
   * @param env The active JavaScript environment.
   * @param exports The exports object to decorate.
   * @return Napi::Object The decorated exports object.
   */
  static Napi::Object Init(Napi::Env env, Napi::Object exports);

  /**
   * @brief Construct a new LoggingResourceAdapter instance from JavaScript.
   *
   */
  LoggingResourceAdapter(CallbackArgs const& args);

  /**
   * @brief Check whether an Napi value is an instance of `LoggingResourceAdapter`.
   *
   * @param val The Napi::Value to test
   * @return true if the value is a `LoggingResourceAdapter`
   * @return false if the value is not a `LoggingResourceAdapter`
   */
  inline static bool is_instance(Napi::Value const& val) {
    return val.IsObject() and val.As<Napi::Object>().InstanceOf(constructor.Value());
  }

  rmm::mr::logging_resource_adaptor<rmm::mr::device_memory_resource>* get_log_mr() {
    return static_cast<rmm::mr::logging_resource_adaptor<rmm::mr::device_memory_resource>*>(
      mr_.get());
  }

  /**
   * @copydoc rmm::mr::logging_resource_adaptor::flush()
   */
  void flush() { get_log_mr()->flush(); }

  std::string const& get_file_path() { return log_file_path_; };

 private:
  static Napi::FunctionReference constructor;

  Napi::Value flush(Napi::CallbackInfo const& info);
  Napi::Value get_file_path(Napi::CallbackInfo const& info);
  Napi::Value get_upstream_mr(Napi::CallbackInfo const& info);

  std::string log_file_path_{};
  Napi::ObjectReference upstream_mr_{};  ///< The upstream MemoryResource.
};

}  // namespace nv
