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
#include <cuda_runtime.h>
#include <napi.h>
#include <nvrtc.h>

namespace node_cuda {

inline Napi::Error cuError(const Napi::Env& env, CUresult code, const char* file, uint32_t line) {
  const char* name;
  const char* estr;
  CUDAAPI::cuGetErrorName(code, &name);
  CUDAAPI::cuGetErrorString(code, &estr);
  auto msg = std::string{name} + " " + std::string{estr} + "\n    at " + std::string{file} + ":" +
             std::to_string(line);
  return Napi::Error::New(env, msg);
}

inline Napi::Error cudaError(const Napi::Env& env,
                             cudaError_t code,
                             const char* file,
                             uint32_t line) {
  const char* name = CUDARTAPI::cudaGetErrorName(code);
  const char* estr = CUDARTAPI::cudaGetErrorString(code);
  auto msg = std::string{name} + " " + std::string{estr} + "\n    at " + std::string{file} + ":" +
             std::to_string(line);
  return Napi::Error::New(env, msg);
}

inline Napi::Error nvrtcError(const Napi::Env& env,
                              nvrtcResult code,
                              const char* file,
                              uint32_t line) {
  const char* name = nvrtcGetErrorString(code);
  auto msg = std::string{name} + "\n    at " + std::string{file} + ":" + std::to_string(line);
  return Napi::Error::New(env, msg);
}

}  // namespace node_cuda
