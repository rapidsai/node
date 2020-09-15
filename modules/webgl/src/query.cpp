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

#include "casting.hpp"
#include "context.hpp"
#include "macros.hpp"

namespace node_webgl {

// GL_EXPORT void glBeginQuery (GLenum target, GLuint id);
Napi::Value WebGL2RenderingContext::BeginQuery(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glBeginQuery(FromJS(info[0]), FromJS(info[1]));
  return env.Undefined();
}

// GL_EXPORT void glGenQueries (GLsizei n, GLuint* ids);
Napi::Value WebGL2RenderingContext::CreateQuery(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GLuint query{};
  GL_EXPORT::glGenQueries(1, &query);
  return WebGLQuery::New(query);
}

// GL_EXPORT void glGenQueries (GLsizei n, GLuint* ids);
Napi::Value WebGL2RenderingContext::CreateQueries(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  std::vector<GLuint> queries(static_cast<size_t>(FromJS(info[0])));
  GL_EXPORT::glGenQueries(queries.size(), queries.data());
  return ToNapi(env)(queries);
}

// GL_EXPORT void glDeleteQueries (GLsizei n, const GLuint* ids);
Napi::Value WebGL2RenderingContext::DeleteQuery(Napi::CallbackInfo const& info) {
  auto env     = info.Env();
  GLuint query = FromJS(info[0]);
  GL_EXPORT::glDeleteQueries(1, &query);
  return env.Undefined();
}

// GL_EXPORT void glDeleteQueries (GLsizei n, const GLuint* ids);
Napi::Value WebGL2RenderingContext::DeleteQueries(Napi::CallbackInfo const& info) {
  auto env                    = info.Env();
  std::vector<GLuint> queries = FromJS(info[0]);
  GL_EXPORT::glDeleteQueries(queries.size(), queries.data());
  return env.Undefined();
}

// GL_EXPORT void glEndQuery (GLenum target);
Napi::Value WebGL2RenderingContext::EndQuery(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glEndQuery(FromJS(info[0]));
  return env.Undefined();
}

// GL_EXPORT void glGetQueryiv (GLenum target, GLenum pname, GLint* params);
Napi::Value WebGL2RenderingContext::GetQuery(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GLint params{};
  GL_EXPORT::glGetQueryiv(FromJS(info[0]), FromJS(info[1]), &params);
  return ToNapi(env)(params);
}

// GL_EXPORT void glGetQueryObjectuiv (GLuint id, GLenum pname, GLuint* params);
Napi::Value WebGL2RenderingContext::GetQueryParameter(Napi::CallbackInfo const& info) {
  auto env      = info.Env();
  GLuint target = FromJS(info[0]);
  GLuint pname  = FromJS(info[1]);
  switch (pname) {
    case GL_QUERY_RESULT: {
      GLuint params{};
      GL_EXPORT::glGetQueryObjectuiv(target, pname, &params);
      return ToNapi(env)(params);
    }
    case GL_QUERY_RESULT_AVAILABLE: {
      GLuint params{};
      GL_EXPORT::glGetQueryObjectuiv(target, pname, &params);
      return ToNapi(env)(static_cast<bool>(params));
    }
  }
  GLEW_THROW(env, GL_INVALID_ENUM);
}

// GL_EXPORT GLboolean glIsQuery (GLuint id);
Napi::Value WebGL2RenderingContext::IsQuery(Napi::CallbackInfo const& info) {
  return ToNapi(info.Env())(GL_EXPORT::glIsQuery(FromJS(info[0])));
}

}  // namespace node_webgl
