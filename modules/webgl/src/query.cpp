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

#include "macros.hpp"
#include "webgl.hpp"

#include <nv_node/utilities/args.hpp>
#include <nv_node/utilities/cpp_to_napi.hpp>

namespace nv {

// GL_EXPORT void glBeginQuery (GLenum target, GLuint id);
Napi::Value WebGL2RenderingContext::BeginQuery(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glBeginQuery(args[0], args[1]);
  return info.Env().Undefined();
}

// GL_EXPORT void glGenQueries (GLsizei n, GLuint* ids);
Napi::Value WebGL2RenderingContext::CreateQuery(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLuint query{};
  GL_EXPORT::glGenQueries(1, &query);
  return WebGLQuery::New(query);
}

// GL_EXPORT void glGenQueries (GLsizei n, GLuint* ids);
Napi::Value WebGL2RenderingContext::CreateQueries(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  std::vector<GLuint> queries(static_cast<size_t>(args[0]));
  GL_EXPORT::glGenQueries(queries.size(), queries.data());
  return CPPToNapi(info)(queries);
}

// GL_EXPORT void glDeleteQueries (GLsizei n, const GLuint* ids);
Napi::Value WebGL2RenderingContext::DeleteQuery(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLuint query      = args[0];
  GL_EXPORT::glDeleteQueries(1, &query);
  return info.Env().Undefined();
}

// GL_EXPORT void glDeleteQueries (GLsizei n, const GLuint* ids);
Napi::Value WebGL2RenderingContext::DeleteQueries(Napi::CallbackInfo const& info) {
  CallbackArgs args           = info;
  std::vector<GLuint> queries = args[0];
  GL_EXPORT::glDeleteQueries(queries.size(), queries.data());
  return info.Env().Undefined();
}

// GL_EXPORT void glEndQuery (GLenum target);
Napi::Value WebGL2RenderingContext::EndQuery(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glEndQuery(args[0]);
  return info.Env().Undefined();
}

// GL_EXPORT void glGetQueryiv (GLenum target, GLenum pname, GLint* params);
Napi::Value WebGL2RenderingContext::GetQuery(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint params{};
  GL_EXPORT::glGetQueryiv(args[0], args[1], &params);
  return CPPToNapi(info)(params);
}

// GL_EXPORT void glGetQueryObjectuiv (GLuint id, GLenum pname, GLuint* params);
Napi::Value WebGL2RenderingContext::GetQueryParameter(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLuint target     = args[0];
  GLuint pname      = args[1];
  switch (pname) {
    case GL_QUERY_RESULT: {
      GLuint params{};
      GL_EXPORT::glGetQueryObjectuiv(target, pname, &params);
      return CPPToNapi(info)(params);
    }
    case GL_QUERY_RESULT_AVAILABLE: {
      GLuint params{};
      GL_EXPORT::glGetQueryObjectuiv(target, pname, &params);
      return CPPToNapi(info)(static_cast<bool>(params));
    }
  }
  GLEW_THROW(info.Env(), GL_INVALID_ENUM);
}

// GL_EXPORT GLboolean glIsQuery (GLuint id);
Napi::Value WebGL2RenderingContext::IsQuery(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  auto is_query     = GL_EXPORT::glIsQuery(args[0]);
  return CPPToNapi(info)(is_query);
}

// GL_EXPORT void glQueryCounter (GLuint id, GLenum target);
Napi::Value WebGL2RenderingContext::QueryCounter(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glQueryCounter(args[0], args[1]);
  return info.Env().Undefined();
}

}  // namespace nv
