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

#include "glfw.hpp"

namespace nv {

namespace detail {
inline std::string glfwGetErrorName(int code) {
  switch (code) {
    case GLFW_NO_ERROR: return "no error";
    case GLFW_NOT_INITIALIZED: return "not initialized";
    case GLFW_NO_CURRENT_CONTEXT: return "no current context";
    case GLFW_INVALID_ENUM: return "invalid enum";
    case GLFW_INVALID_VALUE: return "invalid value";
    case GLFW_OUT_OF_MEMORY: return "out of memory";
    case GLFW_API_UNAVAILABLE: return "api unavailable";
    case GLFW_VERSION_UNAVAILABLE: return "version unavailable";
    case GLFW_PLATFORM_ERROR: return "platform error";
    case GLFW_FORMAT_UNAVAILABLE: return "format unavailable";
    case GLFW_NO_WINDOW_CONTEXT: return "no window context";
    default: return "unknown error";
  }
}
}  // namespace detail

inline Napi::Error glfwError(
  Napi::Env const& env, const int code, const char* err, const char* file, const uint32_t line) {
  auto name = detail::glfwGetErrorName(code);
  auto msg  = std::string{name} + " " + std::string{err} + "\n    at " + std::string{file} + ":" +
             std::to_string(line);
  return Napi::Error::New(env, msg);
}

}  // namespace nv
