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

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <napi.h>

namespace nv {

inline std::runtime_error cuError(CUresult code, std::string const& file, uint32_t line) {
  const char* name;
  const char* estr;
  cuGetErrorName(code, &name);
  cuGetErrorString(code, &estr);
  auto msg =
    std::string{name} + " " + std::string{estr} + "\n    at " + file + ":" + std::to_string(line);
  return std::runtime_error(msg);
}

inline std::runtime_error cudaError(cudaError_t code, std::string const& file, uint32_t line) {
  auto const name = cudaGetErrorName(code);
  auto const estr = cudaGetErrorString(code);
  auto const msg =
    std::string{name} + " " + std::string{estr} + "\n    at " + file + ":" + std::to_string(line);
  return std::runtime_error(msg);
}

inline Napi::Error cuError(CUresult code,
                           std::string const& file,
                           uint32_t line,
                           Napi::Env const& env) {
  return Napi::Error::New(env, cuError(code, file, line).what());
}

inline Napi::Error cudaError(cudaError_t code,
                             std::string const& file,
                             uint32_t line,
                             Napi::Env const& env) {
  return Napi::Error::New(env, cudaError(code, file, line).what());
}

}  // namespace nv

#define NODE_CUDA_EXPECT(expr, message)                   \
  do {                                                    \
    if (!(expr)) NAPI_THROW(std::runtime_error(message)); \
  } while (0)

#define NODE_CU_THROW(code, ...) NAPI_THROW(nv::cuError(code, __FILE__, __LINE__, ##__VA_ARGS__))

/**
 * @brief Error checking macro for CUDA driver API functions.
 *
 * Invokes a CUDA driver API function call, if the call does not return
 * CUDA_SUCCESS, throws an exception detailing the CUDA error that occurred.
 *
 **/
#define NODE_CU_TRY(expr, ...)                                            \
  do {                                                                    \
    CUresult const status = (expr);                                       \
    if (status != CUDA_SUCCESS) { NODE_CU_THROW(status, ##__VA_ARGS__); } \
  } while (0)

#define NODE_CUDA_THROW(code, ...) \
  NAPI_THROW(nv::cudaError(code, __FILE__, __LINE__, ##__VA_ARGS__))

/**
 * @brief Error checking macro for CUDA runtime API functions.
 *
 * Invokes a CUDA runtime API function call, if the call does not return
 * cudaSuccess, invokes cudaGetLastError() to clear the error and throws an
 * exception detailing the CUDA error that occurred.
 *
 **/
#define NODE_CUDA_TRY(expr, ...)              \
  do {                                        \
    cudaError_t const status = (expr);        \
    if (status != cudaSuccess) {              \
      cudaGetLastError();                     \
      NODE_CUDA_THROW(status, ##__VA_ARGS__); \
    }                                         \
  } while (0)
