// Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

namespace nv {

Napi::FunctionReference GraphCOO::constructor;

Napi::Object GraphCOO::Init(Napi::Env env, Napi::Object exports) {
  const Napi::Function ctor = DefineClass(env,
                                          "GraphCOO",
                                          {
                                            InstanceAccessor<&GraphCOO::num_edges>("numEdges"),
                                            InstanceAccessor<&GraphCOO::num_nodes>("numNodes"),
                                            InstanceMethod<&GraphCOO::force_atlas2>("forceAtlas2"),
                                          });
  GraphCOO::constructor     = Napi::Persistent(ctor);
  GraphCOO::constructor.SuppressDestruct();
  exports.Set("GraphCOO", ctor);
  return exports;
}

ObjectUnwrap<GraphCOO> GraphCOO::New(nv::Column const& src, nv::Column const& dst) {
  return constructor.New({src.Value(), dst.Value()});
}

GraphCOO::GraphCOO(CallbackArgs const& args) : Napi::ObjectWrap<GraphCOO>(args) {
  Napi::Object const src          = args[0];
  Napi::Object const dst          = args[1];
  NapiToCPP::Object const options = args[2];

  NODE_CUDA_EXPECT(
    Column::is_instance(src), "GraphCOO requires src argument to a Column", args.Env());
  NODE_CUDA_EXPECT(
    Column::is_instance(dst), "GraphCOO requires dst argument to a Column", args.Env());

  src_            = Napi::Persistent(src);
  dst_            = Napi::Persistent(dst);
  directed_edges_ = options.Get("directedEdges");
}

void GraphCOO::Finalize(Napi::Env env) {}

ValueWrap<size_t> GraphCOO::num_nodes() {
  if (!node_count_computed_) {
    auto const& src      = *Column::Unwrap(src_.Value());
    auto const& dst      = *Column::Unwrap(dst_.Value());
    auto src_max         = src.minmax().second->get_value().ToNumber();
    auto dst_max         = dst.minmax().second->get_value().ToNumber();
    node_count_          = 1 + std::max<int32_t>(src_max, dst_max);
    node_count_computed_ = true;
  }
  return {Env(), node_count_};
}

ValueWrap<size_t> GraphCOO::num_edges() {
  if (!edge_count_computed_) {
    auto const& dst      = *Column::Unwrap(src_.Value());
    auto const& src      = *Column::Unwrap(dst_.Value());
    edge_count_          = directed_edges_ ? src.size() : src[src >= dst]->size();
    edge_count_computed_ = true;
  }
  return {Env(), edge_count_};
}

cugraph::GraphCOOView<int32_t, int32_t, float> GraphCOO::view() {
  auto src = Column::Unwrap(src_.Value())->mutable_view();
  auto dst = Column::Unwrap(dst_.Value())->mutable_view();
  return cugraph::GraphCOOView<int32_t, int32_t, float>(src.begin<int32_t>(),
                                                        dst.begin<int32_t>(),
                                                        nullptr,  // edge_weights
                                                        num_nodes(),
                                                        num_edges());
}

Napi::Value GraphCOO::num_nodes(Napi::CallbackInfo const& info) { return num_nodes(); }

Napi::Value GraphCOO::num_edges(Napi::CallbackInfo const& info) { return num_edges(); }

}  // namespace nv
