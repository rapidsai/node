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

#include <napi.h>
#include <cstddef>
#include <cugraph/graph.hpp>
#undef CUDA_TRY

#include <node_cuda/utilities/error.hpp>
#include <node_cuda/utilities/napi_to_cpp.hpp>

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
  Napi::Object const& src = args[0];
  Napi::Object const& dst = args[1];

  // TODO: (bev) type check

  switch (args.Length()) {
    // TODO: mediate New / JS with zero args
    case 0: break;
    case 2: Initialize(src, dst); break;
    case 3: Initialize(src, dst, args[2]); break;
    case 4: Initialize(src, dst, args[2], args[3]); break;
    default:
      NODE_CUDA_EXPECT(false,
                       "GraphCOO constructor requires a numeric number of vertices and edges, "
                       "and optional stream and memory_resource arguments");
      break;
  }
}

Napi::Object GraphCOO::New(nv::Column const& src,
                           nv::Column const& dst,
                           cudaStream_t stream,
                           rmm::mr::device_memory_resource* mr) {
  const auto inst = GraphCOO::constructor.New({});

  GraphCOO::Unwrap(inst)->Initialize(src.Value(), dst.Value(), stream, mr);
  return inst;
}

void GraphCOO::Initialize(Napi::Object const& src,
                          Napi::Object const& dst,
                          cudaStream_t stream,
                          rmm::mr::device_memory_resource* mr) {
  src_.Reset(src, 1);
  dst_.Reset(dst, 1);
}

void GraphCOO::Finalize(Napi::Env env) {}

size_t GraphCOO::NumberOfNodes() const {
  auto& src = *nv::Column::Unwrap(src_.Value());
  auto& dst = *nv::Column::Unwrap(dst_.Value());

  int* src_indices = reinterpret_cast<int*>(src.data().data());
  int* dst_indices = reinterpret_cast<int*>(dst.data().data());

  size_t N = src.size();

  // TODO: (BEV) all this assumes does not need re-numbering

  auto src_mm = cudf::minmax(src.view());
  auto dst_mm = cudf::minmax(dst.view());

  using ScalarType = cudf::scalar_type_t<int>;
  int src_max      = static_cast<ScalarType*>(src_mm.second.get())->value();
  int dst_max      = static_cast<ScalarType*>(dst_mm.second.get())->value();

  return 1 + std::max(src_max, dst_max);
}

size_t GraphCOO::NumberOfEdges() const {
  // TODO (bev) most of Python version still to be ported

  auto& src = *nv::Column::Unwrap(src_.Value());
  return src.size();
}

cugraph::GraphCOOView<int, int, float> GraphCOO::View() const {
  auto edge_list = this->Value().Get("edgelist").ToObject();

  auto& src = *nv::Column::Unwrap(src_.Value());
  auto& dst = *nv::Column::Unwrap(dst_.Value());

  int* src_indices = reinterpret_cast<int*>(src.data().data());
  int* dst_indices = reinterpret_cast<int*>(src.data().data());
  float* weights   = nullptr;

  // TODO: assumes does not need re-numbering
  int n_verts = NumberOfNodes();
  int n_edges = src.size();
  return cugraph::GraphCOOView<int, int, float>(
    src_indices, dst_indices, weights, n_verts, n_edges);
}

Napi::Value GraphCOO::numberOfNodes(Napi::CallbackInfo const& info) {
  return Napi::Number::New(info.Env(), NumberOfNodes());
}

Napi::Value GraphCOO::numberOfEdges(Napi::CallbackInfo const& info) {
  return Napi::Number::New(info.Env(), NumberOfEdges());
}

}  // namespace nv