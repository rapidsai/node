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

#include <cudf/types.hpp>

#include <napi.h>

#include <algorithm>
#include <cstddef>

namespace nv {

Napi::FunctionReference GraphCOO::constructor;

Napi::Object GraphCOO::Init(Napi::Env env, Napi::Object exports) {
  const Napi::Function ctor =
    DefineClass(env,
                "GraphCOO",
                {
                  InstanceAccessor("numEdges", &GraphCOO::num_edges, nullptr, napi_enumerable),
                  InstanceAccessor("numNodes", &GraphCOO::num_nodes, nullptr, napi_enumerable),
                });
  GraphCOO::constructor = Napi::Persistent(ctor);
  GraphCOO::constructor.SuppressDestruct();
  exports.Set("GraphCOO", ctor);
  return exports;
}

ObjectUnwrap<GraphCOO> GraphCOO::New(nv::Column const& src, nv::Column const& dst) {
  return constructor.New({src.Value(), dst.Value()});
}

GraphCOO::GraphCOO(CallbackArgs const& args) : Napi::ObjectWrap<GraphCOO>(args) {
  Napi::Object const& src = args[0];
  Napi::Object const& dst = args[1];

  NODE_CUDA_EXPECT(Column::is_instance(src), "GraphCOO requires src argument to a Column");
  NODE_CUDA_EXPECT(Column::is_instance(dst), "GraphCOO requires dst argument to a Column");

  src_.Reset(src, 1);
  dst_.Reset(dst, 1);
}

void GraphCOO::Finalize(Napi::Env env) {}

ValueWrap<size_t> GraphCOO::num_nodes() {
  if (!node_count_computed_) {
    using ScalarType = cudf::scalar_type_t<int32_t>;

    Scalar const& src_max = src_column()->minmax().second;
    Scalar const& dst_max = dst_column()->minmax().second;

    node_count_          = 1 + std::max(static_cast<ScalarType*>(src_max)->value(),
                               static_cast<ScalarType*>(dst_max)->value());
    node_count_computed_ = true;
  }
  return {Env(), node_count_};
}

ValueWrap<size_t> GraphCOO::num_edges() {
  if (!edge_count_computed_) {
    Column const& src    = *src_column();
    Column const& dst    = *dst_column();
    edge_count_          = src[src >= dst]->size();
    edge_count_computed_ = true;
  }
  return {Env(), edge_count_};
}

cugraph::GraphCOOView<int32_t, int32_t, float> GraphCOO::View() {
  // TODO: assumes does not need re-numbering
  return cugraph::GraphCOOView<int32_t, int32_t, float>(
    reinterpret_cast<int32_t*>(src_column()->data().data()),
    reinterpret_cast<int32_t*>(dst_column()->data().data()),
    nullptr,  // edge_weights
    num_nodes(),
    num_edges());
}

Napi::Value GraphCOO::num_nodes(Napi::CallbackInfo const& info) { return num_nodes(); }

Napi::Value GraphCOO::num_edges(Napi::CallbackInfo const& info) { return num_edges(); }

}  // namespace nv
