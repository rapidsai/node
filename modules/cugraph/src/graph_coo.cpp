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

Napi::Function GraphCOO::Init(Napi::Env const& env, Napi::Object exports) {
  return DefineClass(env,
                     "GraphCOO",
                     {
                       InstanceAccessor<&GraphCOO::num_edges>("numEdges"),
                       InstanceAccessor<&GraphCOO::num_nodes>("numNodes"),
                       InstanceMethod<&GraphCOO::force_atlas2>("forceAtlas2"),
                     });
}

GraphCOO::wrapper_t GraphCOO::New(Napi::Env const& env,
                                  Column::wrapper_t const& src,
                                  Column::wrapper_t const& dst) {
  return EnvLocalObjectWrap<GraphCOO>::New(env, src, dst);
}

GraphCOO::GraphCOO(CallbackArgs const& args) : EnvLocalObjectWrap<GraphCOO>(args) {
  NODE_CUDA_EXPECT(
    Column::IsInstance(args[0]), "GraphCOO requires src argument to a Column", args.Env());
  NODE_CUDA_EXPECT(
    Column::IsInstance(args[1]), "GraphCOO requires dst argument to a Column", args.Env());

  Column::wrapper_t const src     = args[0];
  Column::wrapper_t const dst     = args[1];
  NapiToCPP::Object const options = args[2];

  src_            = Napi::Persistent(src);
  dst_            = Napi::Persistent(dst);
  directed_edges_ = options.Get("directedEdges");
}

size_t GraphCOO::num_nodes() {
  if (!node_count_computed_) {
    auto const& src      = *src_.Value();
    auto const& dst      = *dst_.Value();
    auto const src_max   = src.minmax().second;
    auto const dst_max   = dst.minmax().second;
    node_count_          = 1 + std::max<int32_t>(  //
                        src_max->get_value().ToNumber(),
                        dst_max->get_value().ToNumber());
    node_count_computed_ = true;
  }
  return node_count_;
}

size_t GraphCOO::num_edges() {
  if (!edge_count_computed_) {
    auto const& src      = *src_.Value();
    auto const& dst      = *dst_.Value();
    edge_count_          = directed_edges_ ? src.size() : src[src >= dst]->size();
    edge_count_computed_ = true;
  }
  return edge_count_;
}

cugraph::GraphCOOView<int32_t, int32_t, float> GraphCOO::view() {
  auto src = src_.Value()->mutable_view();
  auto dst = dst_.Value()->mutable_view();
  return cugraph::GraphCOOView<int32_t, int32_t, float>(src.begin<int32_t>(),
                                                        dst.begin<int32_t>(),
                                                        nullptr,  // edge_weights
                                                        num_nodes(),
                                                        num_edges());
}

Napi::Value GraphCOO::num_nodes(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), num_nodes());
}

Napi::Value GraphCOO::num_edges(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), num_edges());
}

}  // namespace nv
