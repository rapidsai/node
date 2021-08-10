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

#include "types.hpp"
#include "utilities/cpp_to_napi.hpp"

#include <node_cuda/device.hpp>

#include <nv_node/objectwrap.hpp>
#include <nv_node/utilities/args.hpp>

#include <rmm/cuda_stream_view.hpp>
// #include <rmm/mr/device/aligned_resource_adaptor.hpp>
#include <rmm/mr/device/arena_memory_resource.hpp>
#include <rmm/mr/device/binning_memory_resource.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/fixed_size_memory_resource.hpp>
#include <rmm/mr/device/limiting_resource_adaptor.hpp>
#include <rmm/mr/device/logging_resource_adaptor.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>
#include <rmm/mr/device/thread_safe_resource_adaptor.hpp>
#include <rmm/mr/device/thrust_allocator_adaptor.hpp>
#include <rmm/mr/device/tracking_resource_adaptor.hpp>

#include <napi.h>
#include <memory>

namespace nv {

struct MemoryResource : public EnvLocalObjectWrap<MemoryResource> {
  /**
   * @brief Initialize and export the MemoryResource JavaScript constructor and prototype.
   *
   * @param env The active JavaScript environment.
   * @param exports The exports object to decorate.
   * @return Napi::Function The MemoryResource constructor function.
   */
  static Napi::Function Init(Napi::Env const& env, Napi::Object exports);

  inline static wrapper_t Current(Napi::Env const& env) {
    return MemoryResource::Device(env, rmm::cuda_device_id{Device::active_device_id()});
  }

  inline static wrapper_t Device(Napi::Env const& env, rmm::cuda_device_id id) {
    auto resource = Cuda(env);
    auto mr       = rmm::mr::get_per_device_resource(id);
    resource->mr_.reset(mr, [](auto* p) {});
    resource->device_id_ = rmm::cuda_device_id{id.value()};
    resource->type_      = [&](rmm::mr::device_memory_resource* mr) {
      if (mr == nullptr) {
        throw Napi::Error::New(env, "MemoryResource is null");
        // } else if
        // (dynamic_cast<rmm::mr::aligned_resource_adaptor<rmm::mr::device_memory_resource>*>(
        //              mr)) {
        //   return mr_type::aligned_adaptor;
      } else if (dynamic_cast<rmm::mr::arena_memory_resource<rmm::mr::device_memory_resource>*>(
                   mr)) {
        return mr_type::arena;
      } else if (dynamic_cast<rmm::mr::binning_memory_resource<rmm::mr::device_memory_resource>*>(
                   mr)) {
        return mr_type::binning;
      } else if (dynamic_cast<rmm::mr::cuda_async_memory_resource*>(mr)) {
        return mr_type::cuda_async;
      } else if (dynamic_cast<rmm::mr::cuda_memory_resource*>(mr)) {
        return mr_type::cuda;
      } else if (dynamic_cast<
                   rmm::mr::fixed_size_memory_resource<rmm::mr::device_memory_resource>*>(mr)) {
        return mr_type::fixed_size;
      } else if (dynamic_cast<rmm::mr::limiting_resource_adaptor<rmm::mr::device_memory_resource>*>(
                   mr)) {
        return mr_type::limiting_adaptor;
      } else if (dynamic_cast<rmm::mr::logging_resource_adaptor<rmm::mr::device_memory_resource>*>(
                   mr)) {
        return mr_type::logging_adaptor;
      } else if (dynamic_cast<rmm::mr::managed_memory_resource*>(mr)) {
        return mr_type::managed;
      } else if (dynamic_cast<rmm::mr::polymorphic_allocator<rmm::mr::device_memory_resource>*>(
                   mr)) {
        return mr_type::polymorphic_allocator;
      } else if (dynamic_cast<rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>*>(
                   mr)) {
        return mr_type::pool;
      } else if (dynamic_cast<
                   rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>*>(mr)) {
        return mr_type::statistics_adaptor;
      } else if (dynamic_cast<
                   rmm::mr::thread_safe_resource_adaptor<rmm::mr::device_memory_resource>*>(mr)) {
        return mr_type::thread_safe_adaptor;
      } else if (dynamic_cast<rmm::mr::tracking_resource_adaptor<rmm::mr::device_memory_resource>*>(
                   mr)) {
        return mr_type::tracking_adaptor;
      } else if (dynamic_cast<rmm::mr::device_memory_resource*>(mr)) {
        return mr_type::device;
      }

      throw Napi::Error::New(env, std::string{"Unknown MemoryResource type: "} + typeid(mr).name());
    }(mr);

    return resource;
  }

  inline static wrapper_t Cuda(Napi::Env const& env) {
    return EnvLocalObjectWrap<MemoryResource>::New(env, mr_type::cuda);
  }

  inline static wrapper_t Managed(Napi::Env const& env) {
    return EnvLocalObjectWrap<MemoryResource>::New(env, mr_type::managed);
  }

  inline static wrapper_t Pool(Napi::Env const& env,
                               wrapper_t const& upstream_mr,
                               std::size_t initial_pool_size = -1,
                               std::size_t maximum_pool_size = -1) {
    return EnvLocalObjectWrap<MemoryResource>::New(
      env, mr_type::pool, upstream_mr, initial_pool_size, maximum_pool_size);
  }

  inline static wrapper_t FixedSize(Napi::Env const& env,
                                    wrapper_t const& upstream_mr,
                                    std::size_t block_size            = 1 << 20,
                                    std::size_t blocks_to_preallocate = 128) {
    return EnvLocalObjectWrap<MemoryResource>::New(
      env, mr_type::fixed_size, upstream_mr, block_size, blocks_to_preallocate);
  }

  inline static wrapper_t Binning(Napi::Env const& env,
                                  wrapper_t const& upstream_mr,
                                  std::size_t min_size_exponent = -1,
                                  std::size_t max_size_exponent = -1) {
    return EnvLocalObjectWrap<MemoryResource>::New(
      env, mr_type::binning, upstream_mr, min_size_exponent, max_size_exponent);
  }

  inline static wrapper_t Logging(Napi::Env const& env,
                                  wrapper_t const& upstream_mr,
                                  std::string const& log_file_path = "",
                                  bool auto_flush                  = false) {
    return EnvLocalObjectWrap<MemoryResource>::New(
      env, mr_type::logging_adaptor, upstream_mr, log_file_path, auto_flush);
  }

  /**
   * @brief Constructs a new MemoryResource instance.
   *
   */
  MemoryResource(CallbackArgs const& args);

  /**
   * @brief Destructor called when the JavaScript VM garbage collects this MemoryResource
   * instance.
   *
   * @param env The active JavaScript environment.
   */
  void Finalize(Napi::Env env) override;

  inline rmm::cuda_device_id device() const noexcept { return device_id_; }

  inline operator rmm::mr::device_memory_resource*() const { return mr_.get(); }

  std::string file_path() const;

  /**
   * @copydoc rmm::mr::device_memory_resource::is_equal(
   *            rmm::mr::device_memory_resource const& other)
   */
  bool is_equal(Napi::Env const& env, rmm::mr::device_memory_resource const& other) const;

  /**
   * @copydoc rmm::mr::device_memory_resource::get_mem_info(
   *            rmm::cuda_stream_view stream)
   */
  std::pair<std::size_t, std::size_t> get_mem_info(Napi::Env const& env,
                                                   rmm::cuda_stream_view stream) const;

  /**
   * @copydoc rmm::mr::device_memory_resource::supports_streams()
   */
  bool supports_streams(Napi::Env const& env) const;

  /**
   * @copydoc rmm::mr::device_memory_resource::supports_get_mem_info()
   */
  bool supports_get_mem_info(Napi::Env const& env) const;

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
  void add_bin(size_t allocation_size, Napi::Object const& bin_resource);

 private:
  inline rmm::mr::binning_memory_resource<rmm::mr::device_memory_resource>* get_bin_mr() {
    return static_cast<rmm::mr::binning_memory_resource<rmm::mr::device_memory_resource>*>(
      mr_.get());
  }

  inline rmm::mr::logging_resource_adaptor<rmm::mr::device_memory_resource>* get_log_mr() {
    return static_cast<rmm::mr::logging_resource_adaptor<rmm::mr::device_memory_resource>*>(
      mr_.get());
  }

  void flush(Napi::CallbackInfo const& info);
  void add_bin(Napi::CallbackInfo const& info);
  Napi::Value is_equal(Napi::CallbackInfo const& info);

  Napi::Value get_device(Napi::CallbackInfo const& info);
  Napi::Value get_mem_info(Napi::CallbackInfo const& info);
  Napi::Value get_file_path(Napi::CallbackInfo const& info);
  Napi::Value get_upstream_mr(Napi::CallbackInfo const& info);

  Napi::Value supports_streams(Napi::CallbackInfo const& info);
  Napi::Value supports_get_mem_info(Napi::CallbackInfo const& info);

  std::string log_file_path_{};
  mr_type type_{mr_type::cuda};

  Napi::ObjectReference upstream_mr_;
  std::vector<Napi::ObjectReference> bin_mrs_;
  std::shared_ptr<rmm::mr::device_memory_resource> mr_;
  rmm::cuda_device_id device_id_{Device::active_device_id()};
};

}  // namespace nv
