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

#include <node_cuda/utilities/error.hpp>
<<<<<<< HEAD
#include <node_cudf/utilities/napi_to_cpp.hpp>
#include <node_cuml/cuml/manifold/umap.hpp>
#include <node_rmm/device_buffer.hpp>
=======
#include <node_cuda/utilities/napi_to_cpp.hpp>
#include <node_cuml/cuml/manifold/umap.hpp>
>>>>>>> 39ab12b56a16105592323316f2cc72ff5aed6c1b

#include <cuml/manifold/umapparams.h>
#include <cudf/reduction.hpp>
#include <cudf/types.hpp>
#include <cuml/common/device_buffer.hpp>

#include <napi.h>

namespace nv {

  int get_int(NapiToCPP const& opt, int const default_val) {
    return opt.IsNumber() ? opt.operator int() : default_val;
  }

  bool get_bool(NapiToCPP const& opt, bool const default_val) {
    return opt.IsBoolean() ? opt.operator bool() : default_val;
  }

  float get_float(NapiToCPP const& opt, float const default_val) {
    return opt.IsNumber() ? opt.operator float() : default_val;
  }

  Napi::Function UMAP::Init(Napi::Env const& env, Napi::Object exports) {
    return DefineClass(env,
                       "UMAP",
                       {InstanceMethod<&UMAP::fit>("fit"),
                        InstanceMethod<&UMAP::get_embeddings>("getEmbeddings"),
                        InstanceMethod<&UMAP::transform>("transform")});
  }

  ML::UMAPParams update_params(NapiToCPP::Object props) {
    auto params                  = new ML::UMAPParams();
    params->n_neighbors          = get_int(props.Get("n_neighbors"), 15);
    params->n_components         = get_int(props.Get("n_components"), 2);
    params->n_epochs             = get_int(props.Get("n_epochs"), 0);
    params->learning_rate        = get_float(props.Get("learning_rate"), 1.0);
    params->min_dist             = get_float(props.Get("min_dist"), 0.1);
    params->spread               = get_float(props.Get("spread"), 1.0);
    params->set_op_mix_ratio     = get_float(props.Get("set_op_mix_ratio"), 1.0);
    params->local_connectivity   = get_float(props.Get("local_connectivity"), 1.0);
    params->repulsion_strength   = get_float(props.Get("repulsion_strength"), 1.0);
    params->negative_sample_rate = get_int(props.Get("negative_sample_rate"), 5);
    params->transform_queue_size = get_float(props.Get("transform_queue_size"), 4);
    // params->verbosity            = props.Get("verbosity");
    params->a                  = get_float(props.Get("a"), -1.0);
    params->b                  = get_float(props.Get("b"), -1.0);
    params->initial_alpha      = get_float(props.Get("initial_alpha"), 1.0);
    params->init               = get_int(props.Get("init"), 1);
    params->target_n_neighbors = get_int(props.Get("target_n_neighbors"), 1);
    // params->target_metric        = props.Get("target_metric");
    params->target_weights   = get_float(props.Get("target_weights"), 0.5);
    params->random_state     = get_int(props.Get("random_state"), 0);
    params->multicore_implem = get_bool(props.Get("multicore_implem"), true);
    params->optim_batch_size = get_int(props.Get("optim_batch_size"), 0);
    return *params;
  }

  UMAP::wrapper_t UMAP::New(Napi::Env const& env) { return EnvLocalObjectWrap<UMAP>::New(env); }

  UMAP::UMAP(CallbackArgs const& args) : EnvLocalObjectWrap<UMAP>(args) {
    auto env = args.Env();

    NODE_CUDA_EXPECT(args.IsConstructCall(), "UMAP constructor requires 'new'", env);
    // NODE_CUDA_EXPECT(args[0].IsObject(), "UMAP constructor requires a properties Object",
    // env);
    this->params_     = update_params(args[0]);
    this->embeddings_ = rmm::device_buffer();
  }

  void UMAP::fit(float* X,
                 cudf::size_type n_samples,
                 cudf::size_type n_features,
                 float* y,
                 int64_t* knn_indices,
                 float* knn_dists,
                 bool convert_dtype) {
    raft::handle_t handle;
    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));
    this->embeddings_.resize(n_samples * this->params_.n_components, handle.get_stream());
    ML::UMAP::fit(handle,
                  X,
                  y,
                  n_samples,
                  n_features,
                  knn_indices,
                  knn_dists,
                  &this->params_,
                  reinterpret_cast<float*>(this->embeddings_.data()));
  }

  void UMAP::transform(float* X,
                       cudf::size_type n_samples,
                       cudf::size_type n_features,
                       int64_t* knn_indices,
                       float* knn_dists,
                       float* orig_X,
                       int orig_n,
                       bool convert_dtype) {
    raft::handle_t handle;
    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));
    auto transformed = rmm::device_buffer();
    transformed.resize(n_samples * this->params_.n_components, handle.get_stream());

    ML::UMAP::transform(handle,
                        X,
                        n_samples,
                        n_features,
                        knn_indices,
                        knn_dists,
                        orig_X,
                        orig_n,
                        reinterpret_cast<float*>(this->embeddings_.data()),
                        n_samples,
                        &this->params_,
                        reinterpret_cast<float*>(transformed.data()));
  }

  Napi::Value UMAP::fit(Napi::CallbackInfo const& info) {
    CallbackArgs args{info};

    auto X           = DeviceBuffer::wrapper_t(*DeviceBuffer::Unwrap(args[0]));
    auto y           = DeviceBuffer::IsInstance(args[3])
                         ? reinterpret_cast<float*>(DeviceBuffer::Unwrap(args[3])->data())
                         : nullptr;
    auto knn_indices = DeviceBuffer::IsInstance(args[4])
                         ? reinterpret_cast<int64_t*>(DeviceBuffer::Unwrap(args[4])->data())
                         : nullptr;
    auto knn_dists   = DeviceBuffer::IsInstance(args[5])
                         ? reinterpret_cast<float*>(DeviceBuffer::Unwrap(args[5])->data())
                         : nullptr;

    fit(reinterpret_cast<float*>(X->data()), args[1], args[2], y, knn_indices, knn_dists, args[6]);

    return Napi::Value::From(info.Env(), info.Env().Undefined());
  }

  Napi::Value UMAP::transform(Napi::CallbackInfo const& info) {
    CallbackArgs args{info};

    auto X           = DeviceBuffer::wrapper_t(*DeviceBuffer::Unwrap(args[0]));
    auto knn_indices = DeviceBuffer::IsInstance(args[3])
                         ? reinterpret_cast<int64_t*>(DeviceBuffer::Unwrap(args[3])->data())
                         : nullptr;
    auto knn_dists   = DeviceBuffer::IsInstance(args[4])
                         ? reinterpret_cast<float*>(DeviceBuffer::Unwrap(args[4])->data())
                         : nullptr;

    transform(reinterpret_cast<float*>(X->data()),
              args[1],
              args[2],
              knn_indices,
              knn_dists,
              reinterpret_cast<float*>(X->data()),
              args[1],
              args[5]);

    return DeviceBuffer::wrapper_t(
      DeviceBuffer::New(info.Env(), std::make_unique<rmm::device_buffer>(this->get_embeddings())));
    ;
  }

  Napi::Value UMAP::get_embeddings(Napi::CallbackInfo const& info) {
    return DeviceBuffer::wrapper_t(
      DeviceBuffer::New(info.Env(), std::make_unique<rmm::device_buffer>(this->get_embeddings())));
  }

}  // namespace nv
