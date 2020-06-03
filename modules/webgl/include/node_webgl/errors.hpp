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

#include <napi.h>
#include <node_webgl/gl.hpp>

namespace node_webgl {

inline Napi::Error glewError(const Napi::Env& env,
                             const int code,
                             const char* file,
                             const uint32_t line) {
  auto error_name   = reinterpret_cast<const char*>(GLEWAPIENTRY::glewGetString(code));
  auto error_string = reinterpret_cast<const char*>(GLEWAPIENTRY::glewGetErrorString(code));
  auto msg          = std::to_string(code);
  msg += (error_name != NULL ? " " + std::string{error_name} : "");
  msg += (error_string != NULL ? " " + std::string{error_string} : "");
  msg += (file != NULL ? "\n    at " + std::string{file} : "\n");
  return Napi::Error::New(env, msg + ":" + std::to_string(line));
}

}  // namespace node_webgl
