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

struct Graph : public EnvLocalObjectWrap<Graph> {
  /**
   * @brief Initialize and export the Graph JavaScript constructor and prototype.
   *
   * @param env The active JavaScript environment.
   * @param exports The exports object to decorate.
   * @return Napi::Function The Graph constructor function.
   */
  static Napi::Function Init(Napi::Env const& env, Napi::Object exports);

  /**
   * @brief Construct a new Graph instance from C++.
   *
   * @param  src The source node indices for edges
   * @param  dst The destination node indices for edges
   */
  static wrapper_t New(Napi::Env const& env,
                       Column::wrapper_t const& src,
                       Column::wrapper_t const& dst);

  /**
   * @brief Construct a new Graph instance from JavaScript.
   */
  Graph(CallbackArgs const& args);

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
   * @brief Get a non-owning view of the Graph in COO format
   *
   */
  cugraph::legacy::GraphCOOView<int32_t, int32_t, float> coo_view();

  /**
   * @brief Get a non-owning view of the Graph in CSR format
   *
   */
  cugraph::legacy::GraphCSRView<int32_t, int32_t, float> csr_view();

  /**
   * @brief Conversion operator to get a non-owning view of the Graph in COO format
   *
   */
  inline operator cugraph::legacy::GraphCOOView<int32_t, int32_t, float>() { return coo_view(); }

  /**
   * @brief Conversion operator to get a non-owning view of the Graph in CSR format
   *
   */
  inline operator cugraph::legacy::GraphCSRView<int32_t, int32_t, float>() { return csr_view(); }

 private:
  bool directed_edges_{false};

  cudf::size_type edge_count_{};
  bool edge_count_computed_{false};

  cudf::size_type node_count_{};
  bool node_count_computed_{false};

  // edge list columns
  Napi::Reference<Column::wrapper_t> src_;
  Napi::Reference<Column::wrapper_t> dst_;
  Napi::Reference<Column::wrapper_t> e_weights_;

  // adjacency list columns
  Napi::Reference<Column::wrapper_t> offsets_;
  Napi::Reference<Column::wrapper_t> indices_;
  Napi::Reference<Column::wrapper_t> a_weights_;

  Napi::Value num_edges(Napi::CallbackInfo const& info);
  Napi::Value num_nodes(Napi::CallbackInfo const& info);
  Napi::Value degree(Napi::CallbackInfo const& info);

  // layout/force_atlas2.cpp
  Napi::Value force_atlas2(Napi::CallbackInfo const& info);

  // community/spectral_clustering.cpp
  Napi::Value spectral_balanced_cut_clustering(Napi::CallbackInfo const& info);
  Napi::Value spectral_modularity_maximization_clustering(Napi::CallbackInfo const& info);
  Napi::Value analyze_modularity_clustering(Napi::CallbackInfo const& info);
  Napi::Value analyze_edge_cut_clustering(Napi::CallbackInfo const& info);
  Napi::Value analyze_ratio_cut_clustering(Napi::CallbackInfo const& info);
};

}  // namespace nv
