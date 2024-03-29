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
#include "glfw.hpp"

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

#define GLFW_THROW(env, code, err) \
  NAPI_THROW(nv::glfwError(env, code, err, __FILE__, __LINE__), (e).Undefined())

#define GLFW_THROW_ASYNC(task, code, err)                                              \
  (task)->Reject(nv::glfwError((task)->Env(), code, err, __FILE__, __LINE__).Value()); \
  return (task)->Promise()

#define GLFW_TRY(env, expr)                                    \
  do {                                                         \
    (expr);                                                    \
    const char* err = NULL;                                    \
    int const code  = GLFWAPI::glfwGetError(&err);             \
    if (code != GLFW_NO_ERROR) { GLFW_THROW(env, code, err); } \
  } while (0)

#define GLFW_TRY_VOID(env, expr)                   \
  do {                                             \
    (expr);                                        \
    const char* err = NULL;                        \
    int const code  = GLFWAPI::glfwGetError(&err); \
    if (code != GLFW_NO_ERROR) { return; }         \
  } while (0)

#define GLFW_TRY_ASYNC(env, expr)                                    \
  do {                                                               \
    (expr);                                                          \
    const char* err = NULL;                                          \
    int const code  = GLFWAPI::glfwGetError(&err);                   \
    if (code != GLFW_NO_ERROR) { GLFW_THROW_ASYNC(env, code, err); } \
  } while (0)

#define GLFW_EXPECT_TRUE(env, expr)                  \
  do {                                               \
    if ((expr) != GLFW_TRUE) {                       \
      const char* err = NULL;                        \
      int const code  = GLFWAPI::glfwGetError(&err); \
      GLFW_THROW(env, code, err);                    \
    }                                                \
  } while (0)
