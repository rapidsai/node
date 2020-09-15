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
#include <nvEncodeAPI.h>
#include <stdexcept>

namespace nv {

inline std::runtime_error cuError(CUresult code, std::string const& file, uint32_t line) {
  const char* name;
  const char* estr;
  CUDAAPI::cuGetErrorName(code, &name);
  CUDAAPI::cuGetErrorString(code, &estr);
  auto msg =
    std::string{name} + " " + std::string{estr} + "\n    at " + file + ":" + std::to_string(line);
  return std::runtime_error(msg);
}

inline std::runtime_error cudaError(cudaError_t code, std::string const& file, uint32_t line) {
  auto const name = CUDARTAPI::cudaGetErrorName(code);
  auto const estr = CUDARTAPI::cudaGetErrorString(code);
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

inline static char const* nvencGetErrorName(NVENCSTATUS code) {
  static const std::vector<char const*> nvenc_status_code_names{
    "NV_ENC_SUCCESS",
    "NV_ENC_ERR_NO_ENCODE_DEVICE",
    "NV_ENC_ERR_UNSUPPORTED_DEVICE",
    "NV_ENC_ERR_INVALID_ENCODERDEVICE",
    "NV_ENC_ERR_INVALID_DEVICE",
    "NV_ENC_ERR_DEVICE_NOT_EXIST",
    "NV_ENC_ERR_INVALID_PTR",
    "NV_ENC_ERR_INVALID_EVENT",
    "NV_ENC_ERR_INVALID_PARAM",
    "NV_ENC_ERR_INVALID_CALL",
    "NV_ENC_ERR_OUT_OF_MEMORY",
    "NV_ENC_ERR_ENCODER_NOT_INITIALIZED",
    "NV_ENC_ERR_UNSUPPORTED_PARAM",
    "NV_ENC_ERR_LOCK_BUSY",
    "NV_ENC_ERR_NOT_ENOUGH_BUFFER",
    "NV_ENC_ERR_INVALID_VERSION",
    "NV_ENC_ERR_MAP_FAILED",
    "NV_ENC_ERR_NEED_MORE_INPUT",
    "NV_ENC_ERR_ENCODER_BUSY",
    "NV_ENC_ERR_EVENT_NOT_REGISTERD",
    "NV_ENC_ERR_GENERIC",
    "NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY",
    "NV_ENC_ERR_UNIMPLEMENTED",
    "NV_ENC_ERR_RESOURCE_REGISTER_FAILED",
    "NV_ENC_ERR_RESOURCE_NOT_REGISTERED",
    "NV_ENC_ERR_RESOURCE_NOT_MAPPED"};
  return nvenc_status_code_names[code];
}

inline std::runtime_error nvencError(NVENCSTATUS code,
                                     std::string const& estr,
                                     std::string const& file,
                                     uint32_t const line) {
  auto const name = nvencGetErrorName(code);
  auto const msg = std::string{name} + " " + estr + "\n    at " + file + ":" + std::to_string(line);
  return std::runtime_error(msg);
}

inline Napi::Error nvencError(NVENCSTATUS code,
                              std::string const& estr,
                              std::string const& file,
                              uint32_t const line,
                              Napi::Env const& env) {
  auto const name = nvencGetErrorName(code);
  auto const msg = std::string{name} + " " + estr + "\n    at " + file + ":" + std::to_string(line);
  return Napi::Error::New(env, msg);
}

}  // namespace nv
