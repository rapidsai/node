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

// GL_EXPORT void glCreateVertexArrays (GLsizei n, GLuint* arrays);
Napi::Value WebGL2RenderingContext::CreateVertexArray(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLuint vertex_array{};
  GL_EXPORT::glCreateVertexArrays(1, &vertex_array);
  return WebGLVertexArrayObject::New(vertex_array);
}

// GL_EXPORT void glCreateVertexArrays (GLsizei n, GLuint* arrays);
Napi::Value WebGL2RenderingContext::CreateVertexArrays(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  std::vector<GLuint> vertex_arrays(static_cast<size_t>(args[0]));
  GL_EXPORT::glCreateVertexArrays(vertex_arrays.size(), vertex_arrays.data());
  return CPPToNapi(info.Env())(vertex_arrays);
}

// GL_EXPORT void glBindVertexArray (GLuint array);
Napi::Value WebGL2RenderingContext::BindVertexArray(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glBindVertexArray(args[0]);
  return info.Env().Undefined();
}

// GL_EXPORT void glDeleteVertexArrays (GLsizei n, const GLuint* arrays);
Napi::Value WebGL2RenderingContext::DeleteVertexArray(Napi::CallbackInfo const& info) {
  CallbackArgs args   = info;
  GLuint vertex_array = args[0];
  GL_EXPORT::glDeleteVertexArrays(1, &vertex_array);
  return info.Env().Undefined();
}

// GL_EXPORT void glDeleteVertexArrays (GLsizei n, const GLuint* arrays);
Napi::Value WebGL2RenderingContext::DeleteVertexArrays(Napi::CallbackInfo const& info) {
  CallbackArgs args                 = info;
  std::vector<GLuint> vertex_arrays = args[0];
  GL_EXPORT::glDeleteVertexArrays(vertex_arrays.size(), vertex_arrays.data());
  return info.Env().Undefined();
}

// GL_EXPORT GLboolean glIsVertexArray (GLuint array);
Napi::Value WebGL2RenderingContext::IsVertexArray(Napi::CallbackInfo const& info) {
  CallbackArgs args    = info;
  auto is_vertex_array = GL_EXPORT::glIsVertexArray(args[0]);
  return CPPToNapi(info.Env())(is_vertex_array);
}

}  // namespace nv
