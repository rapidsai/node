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

#define CU_THROW(code, ...) NAPI_THROW(nv::cuError(code, __FILE__, __LINE__, ##__VA_ARGS__))

#define CU_TRY(expr, ...)                                            \
  do {                                                               \
    CUresult const status = (expr);                                  \
    if (status != CUDA_SUCCESS) { CU_THROW(status, ##__VA_ARGS__); } \
  } while (0)

#define CUDA_THROW(code, ...) NAPI_THROW(nv::cudaError(code, __FILE__, __LINE__, ##__VA_ARGS__))

#define CUDA_TRY(expr, ...)              \
  do {                                   \
    cudaError_t const status = (expr);   \
    if (status != cudaSuccess) {         \
      cudaGetLastError();                \
      CUDA_THROW(status, ##__VA_ARGS__); \
    }                                    \
  } while (0)

// #define CUDA_TRY_VOID(expr, ...)       \
//   do {                                 \
//     cudaError_t const status = (expr); \
//     if (status != cudaSuccess) {       \
//       cudaGetLastError();              \
//       return;                          \
//     }                                  \
//   } while (0)

#define CUDA_TRY_ASYNC(task, expr)     \
  do {                                 \
    cudaError_t const status = (expr); \
    if (status != cudaSuccess) {       \
      cudaGetLastError();              \
      CUDA_THROW_ASYNC(task, status);  \
    }                                  \
  } while (0)

#define CUDA_THROW_ASYNC(task, status)                                    \
  auto t = (task);                                                        \
  t->Reject(nv::cudaError(status, __FILE__, __LINE__, t->Env()).Value()); \
  return t->Promise()

// #define NVENC_THROW(e, c, m) \
//   NAPI_THROW(nv::nvencError(e, c, m, __FILE__, __LINE__), (e).Undefined())

// #define NVENC_TRY(env, expr, message)                                    \
//   do {                                                                   \
//     NVENCAPI::NVENCSTATUS const status = (expr);                         \
//     if (status != NV_ENC_SUCCESS) { NVENC_THROW(env, status, message); } \
//   } while (0)

#define NVENC_THROW(code, message, ...) \
  NAPI_THROW(nv::nvencError(code, message, __FILE__, __LINE__, ##__VA_ARGS__))

#define NVENC_TRY(expr, message, ...)                                              \
  do {                                                                             \
    NVENCSTATUS const status = (expr);                                             \
    if (status != NV_ENC_SUCCESS) { NVENC_THROW(status, message, ##__VA_ARGS__); } \
  } while (0)
