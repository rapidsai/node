// Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <node_cugraph/cugraph/legacy/graph.hpp>

#include <node_cudf/column.hpp>

#include <node_rmm/device_buffer.hpp>
#include <node_rmm/memory_resource.hpp>

#include <nv_node/objectwrap.hpp>
#include <nv_node/utilities/args.hpp>

#include <napi.h>

namespace nv {

struct GraphCOO : public EnvLocalObjectWrap<GraphCOO> {
  /**
   * @brief Initialize and export the GraphCOO JavaScript constructor and prototype.
   *
   * @param env The active JavaScript environment.
   * @param exports The exports object to decorate.
   * @return Napi::Function The GraphCOO constructor function.
   */
  static Napi::Function Init(Napi::Env const& env, Napi::Object exports);

  /**
   * @brief Construct a new GraphCOO instance from C++.
   *
   * @param  src The source node indices for edges
   * @param  dst The destination node indices for edges
   */
  static wrapper_t New(Napi::Env const& env,
                       Column::wrapper_t const& src,
                       Column::wrapper_t const& dst);

  /**
   * @brief Construct a new GraphCOO instance from JavaScript.
   */
  GraphCOO(CallbackArgs const& args);

  /**
   * @brief Get the number of edges in the graph
   *
   */
  int32_t num_edges();

  /**
   * @brief Get the number of nodes in the graph
   *
   */
  int32_t num_nodes();

  /**
   * @brief Conversion operator to get a non-owning view of the GraphCOO
   *
   */
  inline operator cugraph::legacy::GraphCOOView<int32_t, int32_t, float>() { return view(); }

  /**
   * @brief Get a non-owning view of the Graph
   *
   */
  cugraph::legacy::GraphCOOView<int32_t, int32_t, float> view();

 private:
  Napi::Value num_edges(Napi::CallbackInfo const& info);
  Napi::Value num_nodes(Napi::CallbackInfo const& info);
  Napi::Value force_atlas2(Napi::CallbackInfo const& info);

  Napi::Value degree(Napi::CallbackInfo const& info);

  bool directed_edges_{false};

  cudf::size_type edge_count_{};
  bool edge_count_computed_{false};

  cudf::size_type node_count_{};
  bool node_count_computed_{false};

  Napi::Reference<Column::wrapper_t> src_;
  Napi::Reference<Column::wrapper_t> dst_;
};

}  // namespace nv
