// Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <node_cudf/column.hpp>

#include <nv_node/objectwrap.hpp>
#include <nv_node/utilities/args.hpp>

#include <cuml/manifold/umapparams.h>
#include <cuml/manifold/umap.hpp>
#ifdef CUDA_TRY
#undef CUDA_TRY
#endif
#include <raft/handle.hpp>

#include <napi.h>

namespace nv {
/**
 * @brief An owning wrapper around a cuml::manifold::UMAP
 *
 */
struct UMAP : public EnvLocalObjectWrap<UMAP> {
  /**
   * @brief Initialize and export the UMAP JavaScript constructor and prototype
   *
   * @param env The active JavaScript environment
   * @param exports The exports object to decorate
   * @return Napi::Function The UMAP constructor function
   */
  static Napi::Function Init(Napi::Env const& env, Napi::Object exports);

  /**
   * @brief Construct a new UMAP instance from C++.
   */
  static wrapper_t New(Napi::Env const& env);

  /**
   * @brief Construct a new UMAP instance from JavaScript
   *
   * @param args
   */
  UMAP(CallbackArgs const& args);

  inline rmm::device_buffer get_embeddings() const noexcept { return embeddings_; }
  void fit(float* X,
           cudf::size_type n_samples,
           cudf::size_type n_features,
           float* y,
           int64_t* knn_indices,
           float* knn_dists,
           bool convert_dtype = true);

  // void fit_sparse(float* X,
  //                 cudf::size_type n_samples,
  //                 cudf::size_type n_features,
  //                 float* y,
  //                 int64_t* knn_indices,
  //                 float* knn_dists,
  //                 bool convert_dtype = true);

  void transform(float* X,
                 cudf::size_type n_samples,
                 cudf::size_type n_features,
                 int64_t* knn_indices,
                 float* knn_dists,
                 float* orig_X,
                 int orig_n,
                 bool convert_dtype = true);

  // void transform_sparse(float* X,
  //                       cudf::size_type n_samples,
  //                       cudf::size_type n_features,
  //                       float* y,
  //                       int64_t* knn_indices,
  //                       float* knn_dists,
  //                       bool convert_dtype = true);

 private:
  ML::UMAPParams params_{};
  rmm::device_buffer embeddings_;
  Napi::Value fit(Napi::CallbackInfo const& info);
  Napi::Value fit_sparse(Napi::CallbackInfo const& info);
  Napi::Value transform(Napi::CallbackInfo const& info);
  Napi::Value transform_sparse(Napi::CallbackInfo const& info);
  Napi::Value get_embeddings(Napi::CallbackInfo const& info);
};
}  // namespace nv
// #ifdef CUDA_TRY
// #undef CUDA_TRY
// #endif
// #ifdef CHECK_CUDA
// #undef CHECK_CUDA
// #endif

// #ifdef CHECK_CUDA
// #undef CHECK_CUDA
// #endif
// #ifdef CUDA_TRY
// #undef CUDA_TRY
// #endif
