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

// GL_EXPORT void glCreateRenderbuffers (GLsizei n, GLuint* renderbuffers);
Napi::Value WebGL2RenderingContext::CreateRenderbuffer(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLuint renderbuffer{};
  GL_EXPORT::glCreateRenderbuffers(1, &renderbuffer);
  return WebGLRenderbuffer::New(info.Env(), renderbuffer);
}

// GL_EXPORT void glCreateRenderbuffers (GLsizei n, GLuint* renderbuffers);
Napi::Value WebGL2RenderingContext::CreateRenderbuffers(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  std::vector<GLuint> renderbuffers(args[0].operator size_t());
  GL_EXPORT::glCreateRenderbuffers(renderbuffers.size(), renderbuffers.data());
  return CPPToNapi(info)(renderbuffers);
}

// GL_EXPORT void glBindRenderbuffer (GLenum target, GLuint renderbuffer);
void WebGL2RenderingContext::BindRenderbuffer(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glBindRenderbuffer(args[0], args[1]);
}

// GL_EXPORT void glDeleteRenderbuffers (GLsizei n, const GLuint* renderbuffers);
void WebGL2RenderingContext::DeleteRenderbuffer(Napi::CallbackInfo const& info) {
  CallbackArgs args   = info;
  GLuint renderbuffer = args[0];
  GL_EXPORT::glDeleteRenderbuffers(1, &renderbuffer);
}

// GL_EXPORT void glDeleteRenderbuffers (GLsizei n, const GLuint* renderbuffers);
void WebGL2RenderingContext::DeleteRenderbuffers(Napi::CallbackInfo const& info) {
  CallbackArgs args                 = info;
  std::vector<GLuint> renderbuffers = args[0];
  GL_EXPORT::glDeleteRenderbuffers(renderbuffers.size(), renderbuffers.data());
}

// GL_EXPORT void glGetRenderbufferParameteriv (GLenum target, GLenum pname, GLint* params);
Napi::Value WebGL2RenderingContext::GetRenderbufferParameteriv(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint params{};
  GL_EXPORT::glGetRenderbufferParameteriv(args[0], args[1], &params);
  return CPPToNapi(info)(params);
}

// GL_EXPORT GLboolean glIsRenderbuffer (GLuint renderbuffer);
Napi::Value WebGL2RenderingContext::IsRenderbuffer(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  return CPPToNapi(info)(GL_EXPORT::glIsRenderbuffer(args[0]));
}

// GL_EXPORT void glRenderbufferStorage (GLenum target, GLenum internalformat, GLsizei width,
// GLsizei height);
void WebGL2RenderingContext::RenderbufferStorage(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glRenderbufferStorage(args[0], args[1], args[2], args[3]);
}

// GL_EXPORT void glRenderbufferStorageMultisample (GLenum target, GLsizei samples, GLenum
// internalformat, GLsizei width, GLsizei height);
void WebGL2RenderingContext::RenderbufferStorageMultisample(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glRenderbufferStorageMultisample(args[0], args[1], args[2], args[3], args[4]);
}

}  // namespace nv
