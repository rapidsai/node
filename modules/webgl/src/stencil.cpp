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

// GL_EXPORT void glClearStencil (GLint s);
Napi::Value WebGL2RenderingContext::ClearStencil(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glClearStencil(FromJS(info[0]));
  return env.Undefined();
}

// GL_EXPORT void glStencilFunc (GLenum func, GLint ref, GLuint mask);
Napi::Value WebGL2RenderingContext::StencilFunc(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glStencilFunc(FromJS(info[0]), FromJS(info[1]), FromJS(info[2]));
  return env.Undefined();
}

// GL_EXPORT void glStencilFuncSeparate (GLenum frontfunc, GLenum backfunc, GLint ref, GLuint mask);
Napi::Value WebGL2RenderingContext::StencilFuncSeparate(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glStencilFuncSeparate(
    FromJS(info[0]), FromJS(info[1]), FromJS(info[2]), FromJS(info[3]));
  return env.Undefined();
}

// GL_EXPORT void glStencilMask (GLuint mask);
Napi::Value WebGL2RenderingContext::StencilMask(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glStencilMask(FromJS(info[0]));
  return env.Undefined();
}

// GL_EXPORT void glStencilMaskSeparate (GLenum face, GLuint mask);
Napi::Value WebGL2RenderingContext::StencilMaskSeparate(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glStencilMaskSeparate(FromJS(info[0]), FromJS(info[1]));
  return env.Undefined();
}

// GL_EXPORT void glStencilOp (GLenum fail, GLenum zfail, GLenum zpass);
Napi::Value WebGL2RenderingContext::StencilOp(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glStencilOp(FromJS(info[0]), FromJS(info[1]), FromJS(info[2]));
  return env.Undefined();
}

// GL_EXPORT void glStencilOpSeparate (GLenum face, GLenum sfail, GLenum dpfail, GLenum dppass);
Napi::Value WebGL2RenderingContext::StencilOpSeparate(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glStencilOpSeparate(
    FromJS(info[0]), FromJS(info[1]), FromJS(info[2]), FromJS(info[3]));
  return env.Undefined();
}

}  // namespace node_webgl
