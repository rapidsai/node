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

#include <cugraph/graph.hpp>
#undef CUDA_TRY

#include <node_cudf/column.hpp>

#include <node_rmm/device_buffer.hpp>
#include <node_rmm/memory_resource.hpp>

#include <nv_node/utilities/args.hpp>
#include <nv_node/utilities/wrap.hpp>

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
   */
  static ObjectUnwrap<GraphCOO> New(nv::Column const& src, nv::Column const& dst);

  /**
   * @brief Construct a new GraphCOO instance from JavaScript.
   */
  GraphCOO(CallbackArgs const& args);

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
  ValueWrap<size_t> num_edges();

  /**
   * @brief Get the number of nodes in the graph
   *
   */
  ValueWrap<size_t> num_nodes();

  /**
   * @brief Get a non-owning view of the Graph
   *
   */
  cugraph::GraphCOOView<int32_t, int32_t, float> View();

  ObjectUnwrap<Column> src_column() const { return src_.Value(); }
  ObjectUnwrap<Column> dst_column() const { return dst_.Value(); }

 private:
  static Napi::FunctionReference constructor;

  Napi::Value num_edges(Napi::CallbackInfo const& info);
  Napi::Value num_nodes(Napi::CallbackInfo const& info);

  size_t edge_count_{};
  bool edge_count_computed_{false};

  size_t node_count_{};
  bool node_count_computed_{false};

  Napi::ObjectReference src_{};
  Napi::ObjectReference dst_{};
};

}  // namespace nv
