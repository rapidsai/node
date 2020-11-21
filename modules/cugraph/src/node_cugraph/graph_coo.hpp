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

#include <cugraph/graph.hpp>
#undef CUDA_TRY

#include <node_cudf/column.hpp>
#include <nv_node/utilities/args.hpp>

#include <napi.h>

namespace nv {

class GraphCOO : public Napi::ObjectWrap<GraphCOO> {
 public:
  /**
   * @brief Initialize and export the GraphCOO JavaScript constructor and prototype.
   *
   * @param env The active JavaScript environment.
   * @param exports The exports object to decorate.
   * @return Napi::Object The decorated exports object.
   */
  static Napi::Object Init(Napi::Env env, Napi::Object exports);

  /**
   * @brief Construct a new GraphCOO instance from C++.
   *
   * @param  src The source node indices for edges
   * @param  dst The destination node indices for edges
   * @param  has_data Whether or not the class has data, default = False.
   * @param stream CUDA stream on which memory may be allocated if the memory
   * resource supports streams.
   * @param mr Memory resource to use for the device memory allocation.
   */
  static Napi::Object New(
    nv::Column const& src,
    nv::Column const& dst,
    bool has_data,
    cudaStream_t stream                 = 0,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Construct a new GraphCOO instance from JavaScript.
   *
   */
  GraphCOO(Napi::CallbackInfo const& info);

  /**
   * @brief Initialize the GraphCOO instance created by either C++ or JavaScript.
   *
   * @param  src The source node indices for edges
   * @param  dst The source node indices for edges
   * @param  has_data Whether or not the class has data, default = False.
   * @param stream CUDA stream on which memory may be allocated if the memory
   * resource supports streams.
   * @param mr Memory resource to use for the device memory allocation.
   */
  void Initialize(Napi::Object const& src,
                  Napi::Object const& dst,
                  bool has_data,
                  cudaStream_t stream                 = 0,
                  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Destructor called when the JavaScript VM garbage collects this GraphCOO
   * instance.
   *
   * @param env The active JavaScript environment.
   */
  void Finalize(Napi::Env env) override;

  /**
   * @brief Check whether an Napi value is an instance of `GraphCOO`.
   *
   * @param val The Napi::Value to test
   * @return true if the value is a `GraphCOO`
   * @return false if the value is not a `GraphCOO`
   */
  inline static bool is_instance(Napi::Value const& val) {
    return val.IsObject() and val.As<Napi::Object>().InstanceOf(constructor.Value());
  }

  /**
   * @brief Get the number of edges in the graph
   *
   */
  size_t NumberOfEdges() const;

  /**
   * @brief Get the number of nodes in the graph
   *
   */
  size_t NumberOfNodes() const;

  /**
   * @brief Get a non-owning view of the Graph
   *
   */
  cugraph::GraphCOOView<int, int, float> View() const;

 private:
  static Napi::FunctionReference constructor;

  Napi::ObjectReference src_{};
  Napi::ObjectReference dst_{};

  Napi::Value numberOfEdges(Napi::CallbackInfo const& info);
  Napi::Value numberOfNodes(Napi::CallbackInfo const& info);
};

}  // namespace nv