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

#pragma once

#include "errors.hpp"

#define EXPORT_PROP(exports, name, val) exports.Set(name, val);

#define EXPORT_ENUM(env, exports, name, val) \
  EXPORT_PROP(exports, name, Napi::Number::New(env, val));

#define EXPORT_FUNC(env, exports, name, func)                                                   \
  exports.DefineProperty(Napi::PropertyDescriptor::Function(                                    \
    env,                                                                                        \
    exports,                                                                                    \
    Napi::String::New(env, name),                                                               \
    func,                                                                                       \
    static_cast<napi_property_attributes>(napi_writable | napi_enumerable | napi_configurable), \
    nullptr));

#define CUDA_THROW(e, c) NAPI_THROW(nv::cudaError(e, c, __FILE__, __LINE__), (e).Undefined())

#define CUDA_TRY(env, expr)                                 \
  do {                                                      \
    cudaError_t const status = (expr);                      \
    if (status != cudaSuccess) { CUDA_THROW(env, status); } \
  } while (0)

#define CUDA_TRY_VOID(env, expr)           \
  do {                                     \
    cudaError_t const status = (expr);     \
    if (status != cudaSuccess) { return; } \
  } while (0)

#define CUDA_TRY_ASYNC(task, expr)                                 \
  do {                                                             \
    cudaError_t const status = (expr);                             \
    if (status != cudaSuccess) { CUDA_THROW_ASYNC(task, status); } \
  } while (0)

#define CUDA_THROW_ASYNC(task, status)                                                    \
  (task)->Reject(node_rmm::cudaError((task)->Env(), status, __FILE__, __LINE__).Value()); \
  return (task)->Promise()
