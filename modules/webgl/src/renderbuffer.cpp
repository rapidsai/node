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

// GL_EXPORT void glCreateRenderbuffers (GLsizei n, GLuint* renderbuffers);
Napi::Value WebGL2RenderingContext::CreateRenderbuffer(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GLuint renderbuffer{};
  GL_EXPORT::glCreateRenderbuffers(1, &renderbuffer);
  return WebGLRenderbuffer::New(renderbuffer);
}

// GL_EXPORT void glCreateRenderbuffers (GLsizei n, GLuint* renderbuffers);
Napi::Value WebGL2RenderingContext::CreateRenderbuffers(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  std::vector<GLuint> renderbuffers(FromJS(info[0]).operator size_t());
  GL_EXPORT::glCreateRenderbuffers(renderbuffers.size(), renderbuffers.data());
  return ToNapi(env)(renderbuffers);
}

// GL_EXPORT void glBindRenderbuffer (GLenum target, GLuint renderbuffer);
Napi::Value WebGL2RenderingContext::BindRenderbuffer(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glBindRenderbuffer(FromJS(info[0]), FromJS(info[1]));
  return env.Undefined();
}

// GL_EXPORT void glDeleteRenderbuffers (GLsizei n, const GLuint* renderbuffers);
Napi::Value WebGL2RenderingContext::DeleteRenderbuffer(Napi::CallbackInfo const& info) {
  auto env            = info.Env();
  GLuint renderbuffer = FromJS(info[0]);
  GL_EXPORT::glDeleteRenderbuffers(1, &renderbuffer);
  return env.Undefined();
}

// GL_EXPORT void glDeleteRenderbuffers (GLsizei n, const GLuint* renderbuffers);
Napi::Value WebGL2RenderingContext::DeleteRenderbuffers(Napi::CallbackInfo const& info) {
  auto env                          = info.Env();
  std::vector<GLuint> renderbuffers = FromJS(info[0]);
  GL_EXPORT::glDeleteRenderbuffers(renderbuffers.size(), renderbuffers.data());
  return env.Undefined();
}

// GL_EXPORT void glGetRenderbufferParameteriv (GLenum target, GLenum pname, GLint* params);
Napi::Value WebGL2RenderingContext::GetRenderbufferParameteriv(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GLint params{};
  GL_EXPORT::glGetRenderbufferParameteriv(FromJS(info[0]), FromJS(info[1]), &params);
  return ToNapi(env)(params);
}

// GL_EXPORT GLboolean glIsRenderbuffer (GLuint renderbuffer);
Napi::Value WebGL2RenderingContext::IsRenderbuffer(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  return ToNapi(env)(GL_EXPORT::glIsRenderbuffer(FromJS(info[0])));
}

// GL_EXPORT void glRenderbufferStorage (GLenum target, GLenum internalformat, GLsizei width,
// GLsizei height);
Napi::Value WebGL2RenderingContext::RenderbufferStorage(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glRenderbufferStorage(
    FromJS(info[0]), FromJS(info[1]), FromJS(info[2]), FromJS(info[3]));
  return env.Undefined();
}

// GL_EXPORT void glRenderbufferStorageMultisample (GLenum target, GLsizei samples, GLenum
// internalformat, GLsizei width, GLsizei height);
Napi::Value WebGL2RenderingContext::RenderbufferStorageMultisample(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glRenderbufferStorageMultisample(
    FromJS(info[0]), FromJS(info[1]), FromJS(info[2]), FromJS(info[3]), FromJS(info[4]));
  return env.Undefined();
}

}  // namespace node_webgl
