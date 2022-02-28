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

#pragma once

#ifdef CHECK_CUDA
#undef CHECK_CUDA
#endif
#include <cugraph/utilities/error.hpp>
#ifdef CHECK_CUDA
#undef CHECK_CUDA
#endif

#include <napi.h>

namespace nv {

inline cugraph::logic_error cugraphError(std::string const& message,
                                         std::string const& file,
                                         uint32_t line) {
  return cugraph::logic_error("cuGraph failure:\n" + message + "\n    at " + file + ":" +
                              std::to_string(line));
}

inline Napi::Error cugraphError(std::string const& message,
                                std::string const& file,
                                uint32_t line,
                                Napi::Env const& env) {
  return Napi::Error::New(env, cugraphError(message, file, line).what());
}

}  // namespace nv

#ifndef NODE_CUGRAPH_EXPECT
#define NODE_CUGRAPH_EXPECT(expr, message, ...)                                            \
  do {                                                                                     \
    if (!(expr)) NAPI_THROW(nv::cugraphError(message, __FILE__, __LINE__, ##__VA_ARGS__)); \
  } while (0)
#endif

#ifndef NODE_CUGRAPH_THROW
#define NODE_CUGRAPH_THROW(message, ...) \
  NAPI_THROW(nv::cugraphError(message, __FILE__, __LINE__, ##__VA_ARGS__))
#endif

/**
 * @brief Error checking macro for CUDA runtime API functions.
 *
 * Invokes a CUDA runtime API function call, if the call does not return
 * cudaSuccess, invokes cudaGetLastError() to clear the error and throws an
 * exception detailing the CUDA error that occurred.
 *
 **/
#ifndef NODE_CUGRAPH_TRY
#define NODE_CUGRAPH_TRY(expr, ...)              \
  do {                                           \
    cudaError_t const status = (expr);           \
    if (status != cudaSuccess) {                 \
      cudaGetLastError();                        \
      NODE_CUGRAPH_THROW(status, ##__VA_ARGS__); \
    }                                            \
  } while (0)
#endif
