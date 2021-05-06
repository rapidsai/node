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

// GL_EXPORT void glClearStencil (GLint s);
void WebGL2RenderingContext::ClearStencil(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glClearStencil(args[0]);
}

// GL_EXPORT void glStencilFunc (GLenum func, GLint ref, GLuint mask);
void WebGL2RenderingContext::StencilFunc(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glStencilFunc(args[0], args[1], args[2]);
}

// GL_EXPORT void glStencilFuncSeparate (GLenum frontfunc, GLenum backfunc, GLint ref, GLuint mask);
void WebGL2RenderingContext::StencilFuncSeparate(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glStencilFuncSeparate(args[0], args[1], args[2], args[3]);
}

// GL_EXPORT void glStencilMask (GLuint mask);
void WebGL2RenderingContext::StencilMask(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glStencilMask(args[0]);
}

// GL_EXPORT void glStencilMaskSeparate (GLenum face, GLuint mask);
void WebGL2RenderingContext::StencilMaskSeparate(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glStencilMaskSeparate(args[0], args[1]);
}

// GL_EXPORT void glStencilOp (GLenum fail, GLenum zfail, GLenum zpass);
void WebGL2RenderingContext::StencilOp(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glStencilOp(args[0], args[1], args[2]);
}

// GL_EXPORT void glStencilOpSeparate (GLenum face, GLenum sfail, GLenum dpfail, GLenum dppass);
void WebGL2RenderingContext::StencilOpSeparate(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glStencilOpSeparate(args[0], args[1], args[2], args[3]);
}

}  // namespace nv
