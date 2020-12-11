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

#include <node_cugraph/graph_coo.hpp>

#include <node_cuda/utilities/error.hpp>
#include <node_cuda/utilities/napi_to_cpp.hpp>

#include <cudf/reduction.hpp>
#include <cudf/types.hpp>

#include <napi.h>

#include <algorithm>
#include <cstddef>

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

Napi::Object GraphCOO::New(nv::Column const& src, nv::Column const& dst) {
  CPPToNapiValues const args{GraphCOO::constructor.Env()};
  return GraphCOO::constructor.New({src.Value(), dst.Value()});
}

GraphCOO::GraphCOO(CallbackArgs const& args) : Napi::ObjectWrap<GraphCOO>(args) {
  Napi::Value const& src = args[0];
  Napi::Value const& dst = args[1];

  NODE_CUDA_EXPECT(Column::is_instance(src), "GraphCOO requires src argument to a Column");
  NODE_CUDA_EXPECT(Column::is_instance(dst), "GraphCOO requires dst argument to a Column");

  src_.Reset(src.ToObject(), 1);
  dst_.Reset(dst.ToObject(), 1);
}

void GraphCOO::Finalize(Napi::Env env) {}

size_t GraphCOO::NumberOfNodes() {
  if (!node_count_computed_) {
    using ScalarType = cudf::scalar_type_t<int32_t>;

    auto src_max = std::move(src_column().minmax()).second;
    auto dst_max = std::move(dst_column().minmax()).second;

    node_count_ = 1 + std::max(static_cast<ScalarType*>(src_max)->value(),
                               static_cast<ScalarType*>(dst_max)->value());
  }
  return node_count_;
}

size_t GraphCOO::NumberOfEdges() {
  auto distinct_edges = src_column() >= dst_column();

  auto& src = *nv::Column::Unwrap(src_.Value());
  return src.size();
}

cugraph::GraphCOOView<int, int, float> GraphCOO::View() {
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
