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

// GL_EXPORT void glDrawArraysInstanced (GLenum mode, GLint first, GLsizei count, GLsizei
// primcount);
Napi::Value WebGL2RenderingContext::DrawArraysInstanced(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glDrawArraysInstanced(
    FromJS(info[0]), FromJS(info[1]), FromJS(info[2]), FromJS(info[3]));
  return env.Undefined();
}

// GL_EXPORT void glDrawElementsInstanced (GLenum mode, GLsizei count, GLenum type, const void*
// indices, GLsizei primcount);
Napi::Value WebGL2RenderingContext::DrawElementsInstanced(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glDrawElementsInstanced(
    FromJS(info[0]), FromJS(info[1]), FromJS(info[2]), FromJS(info[3]), FromJS(info[4]));
  return env.Undefined();
}

// GLEWAPI void glVertexAttribDivisor (GLuint index, GLuint divisor);
Napi::Value WebGL2RenderingContext::VertexAttribDivisor(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glVertexAttribDivisor(FromJS(info[0]), FromJS(info[1]));
  return env.Undefined();
}

}  // namespace node_webgl
