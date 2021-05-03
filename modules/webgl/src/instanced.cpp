// Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

// GL_EXPORT void glDrawArraysInstanced (GLenum mode, GLint first, GLsizei count, GLsizei
// primcount);
void WebGL2RenderingContext::DrawArraysInstanced(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glDrawArraysInstanced(args[0], args[1], args[2], args[3]);
}

// GL_EXPORT void glDrawElementsInstanced (GLenum mode, GLsizei count, GLenum type, const void*
// indices, GLsizei primcount);
void WebGL2RenderingContext::DrawElementsInstanced(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glDrawElementsInstanced(args[0], args[1], args[2], args[3], args[4]);
}

// GLEWAPI void glVertexAttribDivisor (GLuint index, GLuint divisor);
void WebGL2RenderingContext::VertexAttribDivisor(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glVertexAttribDivisor(args[0], args[1]);
}

}  // namespace nv
