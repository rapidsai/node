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
#include <node_cudf/utilities/napi_to_cpp.hpp>
#include <node_rmm/device_buffer.hpp>

#include <node_cuml/umap.hpp>

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
                     {InstanceAccessor<&UMAP::n_neighbors>("nNeighbors"),
                      InstanceAccessor<&UMAP::n_components>("nComponents"),
                      InstanceAccessor<&UMAP::n_epochs>("nEpochs"),
                      InstanceAccessor<&UMAP::learning_rate>("learningRate"),
                      InstanceAccessor<&UMAP::min_dist>("minDist"),
                      InstanceAccessor<&UMAP::spread>("spread"),
                      InstanceAccessor<&UMAP::set_op_mix_ratio>("setOpMixRatio"),
                      InstanceAccessor<&UMAP::local_connectivity>("localConnectivity"),
                      InstanceAccessor<&UMAP::repulsion_strength>("repulsionStrength"),
                      InstanceAccessor<&UMAP::negative_sample_rate>("negativeSampleRate"),
                      InstanceAccessor<&UMAP::transform_queue_size>("transformQueueSize"),
                      InstanceAccessor<&UMAP::a>("a"),
                      InstanceAccessor<&UMAP::b>("b"),
                      InstanceAccessor<&UMAP::initial_alpha>("initialAlpha"),
                      InstanceAccessor<&UMAP::init>("init"),
                      InstanceAccessor<&UMAP::target_n_neighbors>("targetNNeighbors"),
                      InstanceAccessor<&UMAP::target_weight>("targetWeight"),
                      InstanceAccessor<&UMAP::random_state>("randomState"),
                      InstanceAccessor<&UMAP::verbosity>("verbosity"),
                      InstanceAccessor<&UMAP::target_metric>("targetMetric"),
                      InstanceMethod<&UMAP::fit>("fit"),
                      InstanceMethod<&UMAP::transform>("transform")});
}

ML::UMAPParams update_params(NapiToCPP::Object props) {
  ML::UMAPParams params{};
  params.n_neighbors          = get_int(props.Get("nNeighbors"), 15);
  params.n_components         = get_int(props.Get("nComponents"), 2);
  params.n_epochs             = get_int(props.Get("nEpochs"), 0);
  params.learning_rate        = get_float(props.Get("learningRate"), 1.0);
  params.min_dist             = get_float(props.Get("minDist"), 0.1);
  params.spread               = get_float(props.Get("spread"), 1.0);
  params.set_op_mix_ratio     = get_float(props.Get("setOpMixRatio"), 1.0);
  params.local_connectivity   = get_float(props.Get("localConnectivity"), 1.0);
  params.repulsion_strength   = get_float(props.Get("repulsionStrength"), 1.0);
  params.negative_sample_rate = get_int(props.Get("negativeSampleRate"), 5);
  params.transform_queue_size = get_float(props.Get("transformQueueSize"), 4);
  params.verbosity            = get_int(props.Get("verbosity"), 4);
  params.a                    = get_float(props.Get("a"), -1.0);
  params.b                    = get_float(props.Get("b"), -1.0);
  params.initial_alpha        = get_float(props.Get("initialAlpha"), 1.0);
  params.init                 = get_int(props.Get("init"), 1);
  params.target_n_neighbors   = get_int(props.Get("targetNNeighbors"), 1);
  params.target_metric        = (get_int(props.Get("targetMetric"), 0) == 0)
                                  ? ML::UMAPParams::MetricType::CATEGORICAL
                                  : ML::UMAPParams::MetricType::EUCLIDEAN;
  params.target_weight        = get_float(props.Get("targetWeight"), 0.5);
  params.random_state         = get_int(props.Get("randomState"), 0);
  return params;
}

UMAP::wrapper_t UMAP::New(Napi::Env const& env) { return EnvLocalObjectWrap<UMAP>::New(env); }

UMAP::UMAP(CallbackArgs const& args) : EnvLocalObjectWrap<UMAP>(args) {
  auto env = args.Env();
  raft::handle_t handle;
  NODE_CUDA_EXPECT(args.IsConstructCall(), "UMAP constructor requires 'new'", env);
  // NODE_CUDA_EXPECT(args[0].IsObject(), "UMAP constructor requires a properties Object",
  // env);
  this->params_ = update_params(args[0]);
  ML::UMAP::find_ab(handle, &this->params_);
  this->embeddings_ = DeviceBuffer::wrapper_t();
}

void UMAP::fit(float* X,
               cudf::size_type n_samples,
               cudf::size_type n_features,
               float* y,
               int64_t* knn_indices,
               float* knn_dists,
               bool convert_dtype,
               float* embeddings) {
  raft::handle_t handle;
  CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

  ML::UMAP::fit(
    handle, X, y, n_samples, n_features, knn_indices, knn_dists, &this->params_, embeddings);
}

void UMAP::transform(float* X,
                     cudf::size_type n_samples,
                     cudf::size_type n_features,
                     int64_t* knn_indices,
                     float* knn_dists,
                     float* orig_X,
                     int orig_n,
                     bool convert_dtype,
                     float* embeddings,
                     float* transformed) {
  raft::handle_t handle;
  CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

  ML::UMAP::transform(handle,
                      X,
                      n_samples,
                      n_features,
                      knn_indices,
                      knn_dists,
                      orig_X,
                      orig_n,
                      embeddings,
                      n_samples,
                      &this->params_,
                      transformed);
}

Napi::Value UMAP::fit(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};

  DeviceBuffer::wrapper_t X = args[0];
  DeviceBuffer::wrapper_t y = DeviceBuffer::IsInstance(args[3]) ? DeviceBuffer::wrapper_t(args[3])
                                                                : DeviceBuffer::New(args.Env());
  DeviceBuffer::wrapper_t knn_indices = DeviceBuffer::IsInstance(args[4])
                                          ? DeviceBuffer::wrapper_t(args[4])
                                          : DeviceBuffer::New(args.Env());
  DeviceBuffer::wrapper_t knn_dists   = DeviceBuffer::IsInstance(args[5])
                                          ? DeviceBuffer::wrapper_t(args[5])
                                          : DeviceBuffer::New(args.Env());

  DeviceBuffer::wrapper_t embeddings = args[7];

  fit(static_cast<float*>(X->data()),
      args[1],
      args[2],
      static_cast<float*>(y->data()),
      static_cast<int64_t*>(knn_indices->data()),
      static_cast<float*>(knn_dists->data()),
      args[6],
      static_cast<float*>(embeddings->data()));

  return embeddings;
}

Napi::Value UMAP::transform(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};

  DeviceBuffer::wrapper_t X           = args[0];
  DeviceBuffer::wrapper_t knn_indices = DeviceBuffer::IsInstance(args[3])
                                          ? DeviceBuffer::wrapper_t(args[3])
                                          : DeviceBuffer::New(args.Env());
  DeviceBuffer::wrapper_t knn_dists   = DeviceBuffer::IsInstance(args[4])
                                          ? DeviceBuffer::wrapper_t(args[4])
                                          : DeviceBuffer::New(args.Env());
  DeviceBuffer::wrapper_t embeddings  = args[6];
  DeviceBuffer::wrapper_t transformed = args[7];

  transform(static_cast<float*>(X->data()),
            args[1],
            args[2],
            static_cast<int64_t*>(knn_indices->data()),
            static_cast<float*>(knn_dists->data()),
            static_cast<float*>(X->data()),
            args[1],
            args[5],
            static_cast<float*>(embeddings->data()),
            static_cast<float*>(transformed->data()));

  return transformed;
}

Napi::Value UMAP::n_neighbors(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), params_.n_neighbors);
}

Napi::Value UMAP::n_components(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), params_.n_components);
}

Napi::Value UMAP::n_epochs(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), params_.n_epochs);
}

Napi::Value UMAP::learning_rate(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), params_.learning_rate);
}
Napi::Value UMAP::min_dist(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), params_.min_dist);
}

Napi::Value UMAP::spread(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), params_.spread);
}
Napi::Value UMAP::set_op_mix_ratio(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), params_.set_op_mix_ratio);
}

Napi::Value UMAP::local_connectivity(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), params_.local_connectivity);
}
Napi::Value UMAP::repulsion_strength(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), params_.repulsion_strength);
}

Napi::Value UMAP::negative_sample_rate(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), params_.negative_sample_rate);
}
Napi::Value UMAP::transform_queue_size(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), params_.transform_queue_size);
}

Napi::Value UMAP::a(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), params_.a);
}
Napi::Value UMAP::b(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), params_.b);
}

Napi::Value UMAP::initial_alpha(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), params_.initial_alpha);
}

Napi::Value UMAP::init(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), params_.init);
}
Napi::Value UMAP::target_n_neighbors(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), params_.target_n_neighbors);
}

Napi::Value UMAP::target_weight(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), params_.target_weight);
}

Napi::Value UMAP::random_state(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), params_.random_state);
}

Napi::Value UMAP::verbosity(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), params_.verbosity);
}

Napi::Value UMAP::target_metric(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), (int)(params_.target_metric));
}

}  // namespace nv
