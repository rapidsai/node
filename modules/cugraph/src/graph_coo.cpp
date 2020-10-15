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

#include "node_cugraph/graph_coo.hpp"

#include <cstddef>
#include <cugraph/graph.hpp>
#undef CUDA_TRY

#include <node_cuda/utilities/error.hpp>
#include <node_cuda/utilities/napi_to_cpp.hpp>
#include <node_cudf/column.hpp>

#include <cudf/reduction.hpp>


#include <algorithm>

namespace nv {

Napi::FunctionReference GraphCOO::constructor;

Napi::Object GraphCOO::Init(Napi::Env env, Napi::Object exports) {
  const Napi::Function ctor = DefineClass(
    env,
    "GraphCOO",
    {
      InstanceAccessor("numberOfEdges", &GraphCOO::numberOfEdges, nullptr, napi_enumerable),
      InstanceAccessor("numberOfNodes", &GraphCOO::numberOfNodes, nullptr, napi_enumerable),
    });
  GraphCOO::constructor = Napi::Persistent(ctor);
  GraphCOO::constructor.SuppressDestruct();
  exports.Set("GraphCOO", ctor);
  return exports;
}

GraphCOO::GraphCOO(Napi::CallbackInfo const& info) : Napi::ObjectWrap<GraphCOO>(info) {
  const CallbackArgs args{info};
  int number_of_vertices = args[0];
  int number_of_edges = args[1];
  bool has_data = args[2];

  switch (args.Length()) {
    case 3: Initialize(number_of_vertices, number_of_edges, has_data); break;
    case 4: Initialize(number_of_vertices, number_of_edges, args[3]); break;
    case 5: Initialize(number_of_vertices, number_of_edges, args[3], args[4]); break;
    default:
      NODE_CUDA_EXPECT(false,
                       "GraphCOO constructor requires a numeric number of vertices and edges, "
                       "bool has_data, and optional stream and memory_resource arguments");
      break;
  }
}

Napi::Object GraphCOO::New(int number_of_vertices,
                          int number_of_edges,
                          bool has_data,
                          cudaStream_t stream,
                          rmm::mr::device_memory_resource* mr) {
  const auto inst = GraphCOO::constructor.New({});
  GraphCOO::Unwrap(inst)->Initialize(number_of_vertices, number_of_edges, has_data, stream, mr);
  return inst;
}

void GraphCOO::Initialize(int number_of_vertices,
                          int number_of_edges,
                          bool has_data,
                          cudaStream_t stream,
                          rmm::mr::device_memory_resource* mr) {
  resource_.reset(new cugraph::GraphCOO<int, int, float>(number_of_vertices, number_of_edges, has_data, stream, mr));
                  }

void GraphCOO::Finalize(Napi::Env env) {
  if (resource_ != nullptr) { this->resource_ = nullptr; }
  resource_ = nullptr;
}

size_t GraphCOO::NumberOfNodes() const {
  auto edge_list = this->Value().Get("edgelist").ToObject();

  auto& src = *nv::Column::Unwrap(edge_list.Get("src").ToObject());
  auto& dst = *nv::Column::Unwrap(edge_list.Get("dst").ToObject());

  int* src_indices = reinterpret_cast<int*>(src.data().data());
  int* dst_indices = reinterpret_cast<int*>(dst.data().data());

  size_t N = src.size();

  // TODO: (BEV) all this assumes does not need re-numbering

  auto src_mm = cudf::minmax(src.view());
  auto dst_mm = cudf::minmax(dst.view());

  using ScalarType = cudf::scalar_type_t<int>;
  int src_max = static_cast<ScalarType *>(src_mm.second.get())->value();
  int dst_max = static_cast<ScalarType *>(dst_mm.second.get())->value();

  return 1 + std::max(src_max, dst_max);

}

size_t GraphCOO::NumberOfEdges() const {
  // TODO (bev) most of Python version still to be ported

  auto edge_list = this->Value().Get("edgelist").ToObject();

  auto& src = *nv::Column::Unwrap(edge_list.Get("src").ToObject());
  return src.size();
}

cugraph::GraphCOOView<int, int, float> GraphCOO::View() const {
  auto edge_list = this->Value().Get("edgelist").ToObject();

  auto& src = *nv::Column::Unwrap(edge_list.Get("src").ToObject());
  auto& dst = *nv::Column::Unwrap(edge_list.Get("dst").ToObject());

  int* src_indices = reinterpret_cast<int*>(src.data().data());
  int* dst_indices = reinterpret_cast<int*>(src.data().data());
  float* weights   = nullptr;

  // TODO: assumes does not need re-numbering
  int n_verts = NumberOfNodes();
  int n_edges = src.size();
  return cugraph::GraphCOOView<int, int, float>(src_indices, dst_indices, weights, n_verts, n_edges);
}

Napi::Value GraphCOO::numberOfNodes(Napi::CallbackInfo const& info) {
  return Napi::Number::New(info.Env(), NumberOfNodes());
}

Napi::Value GraphCOO::numberOfEdges(Napi::CallbackInfo const& info) {
  return Napi::Number::New(info.Env(), NumberOfEdges());
}

}  // namespace nv