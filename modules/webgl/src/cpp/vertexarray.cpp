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

#include <node_webgl/casting.hpp>
#include <node_webgl/context.hpp>
#include <node_webgl/macros.hpp>

namespace node_webgl {

// GL_EXPORT void glCreateVertexArrays (GLsizei n, GLuint* arrays);
Napi::Value WebGL2RenderingContext::CreateVertexArray(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GLuint vertex_array{};
  GL_EXPORT::glCreateVertexArrays(1, &vertex_array);
  return WebGLVertexArrayObject::New(vertex_array);
}

// GL_EXPORT void glCreateVertexArrays (GLsizei n, GLuint* arrays);
Napi::Value WebGL2RenderingContext::CreateVertexArrays(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  std::vector<GLuint> vertex_arrays(static_cast<size_t>(FromJS(info[0])));
  GL_EXPORT::glCreateVertexArrays(vertex_arrays.size(), vertex_arrays.data());
  return ToNapi(env)(vertex_arrays);
}

// GL_EXPORT void glBindVertexArray (GLuint array);
Napi::Value WebGL2RenderingContext::BindVertexArray(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glBindVertexArray(FromJS(info[0]));
  return env.Undefined();
}

// GL_EXPORT void glDeleteVertexArrays (GLsizei n, const GLuint* arrays);
Napi::Value WebGL2RenderingContext::DeleteVertexArray(Napi::CallbackInfo const& info) {
  auto env            = info.Env();
  GLuint vertex_array = FromJS(info[0]);
  GL_EXPORT::glDeleteVertexArrays(1, &vertex_array);
  return env.Undefined();
}

// GL_EXPORT void glDeleteVertexArrays (GLsizei n, const GLuint* arrays);
Napi::Value WebGL2RenderingContext::DeleteVertexArrays(Napi::CallbackInfo const& info) {
  auto env                          = info.Env();
  std::vector<GLuint> vertex_arrays = FromJS(info[0]);
  GL_EXPORT::glDeleteVertexArrays(vertex_arrays.size(), vertex_arrays.data());
  return env.Undefined();
}

// GL_EXPORT GLboolean glIsVertexArray (GLuint array);
Napi::Value WebGL2RenderingContext::IsVertexArray(Napi::CallbackInfo const& info) {
  return ToNapi(info.Env())(GL_EXPORT::glIsVertexArray(FromJS(info[0])));
}

}  // namespace node_webgl
