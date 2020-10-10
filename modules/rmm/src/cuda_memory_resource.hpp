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

#include <nv_node/utilities/args.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <napi.h>
#include <memory>

namespace nv {

/**
 * @brief An owning wrapper around a device CudaMemoryResource.
 *
 */
class CudaMemoryResource : public Napi::ObjectWrap<CudaMemoryResource> {
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
   * @brief Construct a new CudaMemoryRssource instance from C++.
   *
   */
  static Napi::Value New();

  /**
   * @brief Construct a new CudaMemoryResource instance from JavaScript.
   *
   */
  CudaMemoryResource(Napi::CallbackInfo const& info);

  /**
   * @brief Initialize the CudaMemoryResource instance created by either C++ or JavaScript.
   *
   */
  void Initialize();

  /**
   * @brief Destructor called when the JavaScript VM garbage collects this CudaMemoryResource
   * instance.
   *
   * @param env The active JavaScript environment.
   */
  void Finalize(Napi::Env env) override;

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

  /**
   * @brief Get a pointer to the underlying `CudaMemoryResource`.
   *
   * @return pointer to `CudaMemoryResource`
   */
  auto const& Resource() const { return resource_; }

 private:
  static Napi::FunctionReference constructor;

  Napi::Value allocate(Napi::CallbackInfo const& info);
  Napi::Value deallocate(Napi::CallbackInfo const& info);
  Napi::Value getMemInfo(Napi::CallbackInfo const& info);
  Napi::Value isEqual(Napi::CallbackInfo const& info);
  Napi::Value supportsStreams(Napi::CallbackInfo const& info);
  Napi::Value supportsGetMemInfo(Napi::CallbackInfo const& info);

  std::shared_ptr<rmm::mr::cuda_memory_resource> resource_;
};

}  // namespace nv
