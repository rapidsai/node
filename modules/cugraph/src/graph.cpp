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

#include <node_cugraph/graph.hpp>
#include <node_cugraph/utilities/error.hpp>

#include <node_cudf/table.hpp>
#include <node_cudf/utilities/dtypes.hpp>

#include <node_cuda/utilities/error.hpp>
#include <node_cuda/utilities/napi_to_cpp.hpp>

#include <cugraph/legacy/functions.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>

#include <napi.h>

namespace nv {

namespace {

template <cudf::type_id type>
Column::wrapper_t get_col(NapiToCPP::Object opts, std::string const& name) {
  auto env = opts.Env();
  auto val = opts.Get(name);

  if (val.IsEmpty() || val.IsNull() || val.IsUndefined()) {
    NODE_CUGRAPH_THROW("Graph requires `" + name + "` to be a Column");
  }

  NODE_CUGRAPH_EXPECT(Column::IsInstance(val), "Graph requires `" + name + "` to be a Column", env);

  Column::wrapper_t col = val.As<Napi::Object>();

  NODE_CUGRAPH_EXPECT(col->type().id() == type,
                      "Graph requires `" + name + "` to be a Column of " +
                        cudf::type_dispatcher(cudf::data_type{type}, cudf::type_to_name{}),
                      env);
  return col;
}

template <typename vertex_t, typename edge_t, typename weight_t>
std::tuple<Column::wrapper_t, Column::wrapper_t, Column::wrapper_t> coo_to_csr(
  Napi::Env env,
  Column::wrapper_t const& src,
  Column::wrapper_t const& dst,
  Column::wrapper_t const& e_weights,
  int32_t const num_nodes,
  int32_t const num_edges) {
  auto csr = cugraph::coo_to_csr(cugraph::legacy::GraphCOOView<vertex_t, edge_t, weight_t>(
                                   src->mutable_view().begin<edge_t>(),
                                   dst->mutable_view().begin<edge_t>(),
                                   e_weights->mutable_view().begin<weight_t>(),
                                   num_nodes,
                                   num_edges))
               ->release();

  auto csr_col = [&](cudf::type_id type_id, rmm::device_buffer& data) {
    auto type = cudf::data_type{type_id};
    auto size = data.size() / cudf::size_of(type);
    return Column::New(env, std::make_unique<cudf::column>(type, size, std::move(data)));
  };

  auto offsets   = csr_col(cudf::type_to_id<edge_t>(), *csr.offsets.release());
  auto indices   = csr_col(cudf::type_to_id<vertex_t>(), *csr.indices.release());
  auto a_weights = csr_col(cudf::type_to_id<weight_t>(), *csr.edge_data.release());

  return std::make_tuple(offsets, indices, a_weights);
}

}  // namespace

Napi::Function Graph::Init(Napi::Env const& env, Napi::Object exports) {
  return DefineClass(
    env,
    "Graph",
    {
      InstanceMethod<&Graph::num_edges>("numEdges"),
      InstanceMethod<&Graph::num_nodes>("numNodes"),
      InstanceMethod<&Graph::force_atlas2>("forceAtlas2"),
      InstanceMethod<&Graph::degree>("degree"),

      InstanceMethod<&Graph::spectral_modularity_maximization_clustering>(
        "spectralModularityMaximizationClustering"),
      InstanceMethod<&Graph::spectral_balanced_cut_clustering>("spectralBalancedCutClustering"),
      InstanceMethod<&Graph::analyze_modularity_clustering>("analyzeModularityClustering"),
      InstanceMethod<&Graph::analyze_edge_cut_clustering>("analyzeEdgeCutClustering"),
      InstanceMethod<&Graph::analyze_ratio_cut_clustering>("analyzeRatioCutClustering"),
    });
}

Graph::wrapper_t Graph::New(Napi::Env const& env,
                            Column::wrapper_t const& src,
                            Column::wrapper_t const& dst) {
  return EnvLocalObjectWrap<Graph>::New(env, src, dst);
}

Graph::Graph(CallbackArgs const& args) : EnvLocalObjectWrap<Graph>(args) {
  NapiToCPP::Object opts = args[0];
  directed_edges_        = opts.Get("directed").ToBoolean();
  src_                   = Napi::Persistent(get_col<cudf::type_id::INT32>(opts, "src"));
  dst_                   = Napi::Persistent(get_col<cudf::type_id::INT32>(opts, "dst"));
  e_weights_             = Napi::Persistent(get_col<cudf::type_id::FLOAT32>(opts, "weight"));
}

int32_t Graph::num_nodes() {
  if (!node_count_computed_) {
    node_count_ =
      1 + std::max(src_.Value()->minmax().second->get_value().ToNumber().Int32Value(),  //
                   dst_.Value()->minmax().second->get_value().ToNumber().Int32Value());
    node_count_computed_ = true;
  }
  return node_count_;
}

int32_t Graph::num_edges() {
  if (!edge_count_computed_) {
    auto const& src      = *src_.Value();
    auto const& dst      = *dst_.Value();
    edge_count_          = directed_edges_ ? src.size() : src[src >= dst]->size();
    edge_count_computed_ = true;
  }
  return edge_count_;
}

cugraph::legacy::GraphCOOView<int32_t, int32_t, float> Graph::coo_view() {
  return cugraph::legacy::GraphCOOView<int32_t, int32_t, float>(
    src_.Value()->mutable_view().begin<int32_t>(),
    dst_.Value()->mutable_view().begin<int32_t>(),
    e_weights_.Value()->mutable_view().begin<float>(),
    num_nodes(),
    num_edges());
}

cugraph::legacy::GraphCSRView<int32_t, int32_t, float> Graph::csr_view() {
  if (offsets_.IsEmpty()) {
    auto csr = coo_to_csr<int32_t, int32_t, float>(
      Env(), src_.Value(), dst_.Value(), e_weights_.Value(), num_nodes(), num_edges());
    offsets_   = Napi::Persistent(std::move(std::get<0>(csr)));
    indices_   = Napi::Persistent(std::move(std::get<1>(csr)));
    a_weights_ = Napi::Persistent(std::move(std::get<2>(csr)));
  }

  return cugraph::legacy::GraphCSRView<int32_t, int32_t, float>(
    offsets_.Value()->mutable_view().begin<int32_t>(),
    indices_.Value()->mutable_view().begin<int32_t>(),
    a_weights_.Value()->mutable_view().begin<float>(),
    num_nodes(),
    num_edges());
}

Napi::Value Graph::num_nodes(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), num_nodes());
}

Napi::Value Graph::num_edges(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), num_edges());
}

Napi::Value Graph::degree(Napi::CallbackInfo const& info) {
  auto degree = Column::zeros(info.Env(), cudf::type_id::INT32, num_nodes());

  coo_view().degree(degree->mutable_view().begin<int32_t>(),
                    cugraph::legacy::DegreeDirection::IN_PLUS_OUT);

  return degree;
}

}  // namespace nv
