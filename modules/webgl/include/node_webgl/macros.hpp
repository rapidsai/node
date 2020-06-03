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
  EXPORT_PROP(exports, name, Napi::Number::New(env, static_cast<int64_t>(val)));

#define EXPORT_FUNC(env, exports, name, func)                                                   \
  exports.DefineProperty(Napi::PropertyDescriptor::Function(                                    \
    env,                                                                                        \
    exports,                                                                                    \
    Napi::String::New(env, name),                                                               \
    func,                                                                                       \
    static_cast<napi_property_attributes>(napi_writable | napi_enumerable | napi_configurable), \
    nullptr));

#define GLEW_THROW(env, code) \
  NAPI_THROW(node_webgl::glewError(env, code, __FILE__, __LINE__), (e).Undefined())

#define GLEW_THROW_ASYNC(task, code)                                                      \
  (task)->Reject(node_webgl::glewError((task)->Env(), code, __FILE__, __LINE__).Value()); \
  return (task)->Promise()

#define GL_TRY(env, expr)                                 \
  do {                                                    \
    (expr);                                               \
    GLenum const code = GL_EXPORT::glGetError();          \
    if (code != GLEW_NO_ERROR) { GLEW_THROW(env, code); } \
  } while (0)

#define GL_TRY_VOID(env, expr)                   \
  do {                                           \
    (expr);                                      \
    GLenum const code = GL_EXPORT::glGetError(); \
    if (code != GLEW_NO_ERROR) { return; }       \
  } while (0)

#define GL_TRY_ASYNC(env, expr)                                 \
  do {                                                          \
    (expr);                                                     \
    GLenum const code = GL_EXPORT::glGetError();                \
    if (code != GLEW_NO_ERROR) { GLEW_THROW_ASYNC(env, code); } \
  } while (0)

#define GL_EXPECT_OK(env, expr)                     \
  do {                                              \
    GLenum const code = (expr);                     \
    if (code != GLEW_OK) { GLEW_THROW(env, code); } \
  } while (0)
