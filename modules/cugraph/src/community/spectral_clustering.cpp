// Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <node_cugraph/cugraph/algorithms.hpp>

#include <node_cugraph/graph.hpp>

#include <node_cudf/column.hpp>

namespace nv {

Napi::Value Graph::spectral_balanced_cut_clustering(Napi::CallbackInfo const& info) {
  NapiToCPP::Object opts = info[0];
  int32_t num_clusters   = opts.Get("num_clusters");
  int32_t num_eigen_vecs = opts.Get("num_eigen_vecs");
  float evs_tolerance    = opts.Get("evs_tolerance");
  int32_t evs_max_iter   = opts.Get("evs_max_iter");
  float kmean_tolerance  = opts.Get("kmean_tolerance");
  int32_t kmean_max_iter = opts.Get("kmean_max_iter");

  auto cluster = Column::zeros(info.Env(), cudf::type_id::INT32, num_nodes());

  try {
    cugraph::ext_raft::balancedCutClustering(csr_view(),
                                             num_clusters,
                                             num_eigen_vecs,
                                             evs_tolerance,
                                             evs_max_iter,
                                             kmean_tolerance,
                                             kmean_max_iter,
                                             cluster->mutable_view().begin<int32_t>());
  } catch (std::exception const& e) { throw Napi::Error::New(info.Env(), e.what()); }

  return cluster;
}

Napi::Value Graph::spectral_modularity_maximization_clustering(Napi::CallbackInfo const& info) {
  NapiToCPP::Object opts = info[0];
  int32_t num_clusters   = opts.Get("num_clusters");
  int32_t num_eigen_vecs = opts.Get("num_eigen_vecs");
  float evs_tolerance    = opts.Get("evs_tolerance");
  int32_t evs_max_iter   = opts.Get("evs_max_iter");
  float kmean_tolerance  = opts.Get("kmean_tolerance");
  int32_t kmean_max_iter = opts.Get("kmean_max_iter");

  auto cluster = Column::zeros(info.Env(), cudf::type_id::INT32, num_nodes());

  try {
    cugraph::ext_raft::spectralModularityMaximization(csr_view(),
                                                      num_clusters,
                                                      num_eigen_vecs,
                                                      evs_tolerance,
                                                      evs_max_iter,
                                                      kmean_tolerance,
                                                      kmean_max_iter,
                                                      cluster->mutable_view().begin<int32_t>());
  } catch (std::exception const& e) { throw Napi::Error::New(info.Env(), e.what()); }

  return cluster;
}

Napi::Value Graph::analyze_modularity_clustering(Napi::CallbackInfo const& info) {
  int32_t num_clusters      = info[0].ToNumber();
  Column::wrapper_t cluster = info[1].ToObject();

  float score{};

  try {
    cugraph::ext_raft::analyzeClustering_modularity(
      csr_view(), num_clusters, cluster->view().begin<int32_t>(), &score);
  } catch (std::exception const& e) { throw Napi::Error::New(info.Env(), e.what()); }

  return Napi::Number::New(info.Env(), score);
}

Napi::Value Graph::analyze_edge_cut_clustering(Napi::CallbackInfo const& info) {
  int32_t num_clusters      = info[0].ToNumber();
  Column::wrapper_t cluster = info[1].ToObject();

  float score{};

  try {
    cugraph::ext_raft::analyzeClustering_edge_cut(
      csr_view(), num_clusters, cluster->view().begin<int32_t>(), &score);
  } catch (std::exception const& e) { throw Napi::Error::New(info.Env(), e.what()); }

  return Napi::Number::New(info.Env(), score);
}

Napi::Value Graph::analyze_ratio_cut_clustering(Napi::CallbackInfo const& info) {
  int32_t num_clusters      = info[0].ToNumber();
  Column::wrapper_t cluster = info[1].ToObject();

  float score{};

  try {
    cugraph::ext_raft::analyzeClustering_ratio_cut(
      csr_view(), num_clusters, cluster->view().begin<int32_t>(), &score);
  } catch (std::exception const& e) { throw Napi::Error::New(info.Env(), e.what()); }

  return Napi::Number::New(info.Env(), score);
}

}  // namespace nv
