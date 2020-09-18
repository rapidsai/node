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

// GL_EXPORT void glBindFramebuffer (GLenum target, GLuint framebuffer);
Napi::Value WebGL2RenderingContext::BindFramebuffer(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glBindFramebuffer(args[0], args[1]);
  return info.Env().Undefined();
}

// GL_EXPORT GLenum glCheckFramebufferStatus (GLenum target);
Napi::Value WebGL2RenderingContext::CheckFramebufferStatus(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  return CPPToNapi(info)(GL_EXPORT::glCheckFramebufferStatus(args[0]));
}

// GL_EXPORT void glCreateFramebuffers (GLsizei n, GLuint* framebuffers);
Napi::Value WebGL2RenderingContext::CreateFramebuffer(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLuint framebuffer{};
  GL_EXPORT::glCreateFramebuffers(1, &framebuffer);
  return WebGLFramebuffer::New(framebuffer);
}

// GL_EXPORT void glCreateFramebuffers (GLsizei n, GLuint* framebuffers);
Napi::Value WebGL2RenderingContext::CreateFramebuffers(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  std::vector<GLuint> framebuffers(args[0].operator size_t());
  GL_EXPORT::glCreateFramebuffers(framebuffers.size(), framebuffers.data());
  return CPPToNapi(info)(framebuffers);
}

// GL_EXPORT void glDeleteFramebuffers (GLsizei n, const GLuint* framebuffers);
Napi::Value WebGL2RenderingContext::DeleteFramebuffer(Napi::CallbackInfo const& info) {
  CallbackArgs args  = info;
  GLuint framebuffer = args[0];
  GL_EXPORT::glDeleteFramebuffers(1, &framebuffer);
  return info.Env().Undefined();
}

// GL_EXPORT void glDeleteFramebuffers (GLsizei n, const GLuint* framebuffers);
Napi::Value WebGL2RenderingContext::DeleteFramebuffers(Napi::CallbackInfo const& info) {
  CallbackArgs args                = info;
  std::vector<GLuint> framebuffers = args[0];
  GL_EXPORT::glDeleteFramebuffers(framebuffers.size(), framebuffers.data());
  return info.Env().Undefined();
}

// GL_EXPORT void glFramebufferRenderbuffer (GLenum target, GLenum attachment, GLenum
// renderbuffertarget, GLuint renderbuffer);
Napi::Value WebGL2RenderingContext::FramebufferRenderbuffer(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glFramebufferRenderbuffer(args[0], args[1], args[2], args[3]);
  return info.Env().Undefined();
}

// GL_EXPORT void glFramebufferTexture2D (GLenum target, GLenum attachment, GLenum textarget, GLuint
// texture, GLint level);
Napi::Value WebGL2RenderingContext::FramebufferTexture2D(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glFramebufferTexture2D(args[0], args[1], args[2], args[3], args[4]);
  return info.Env().Undefined();
}

// GL_EXPORT void glGetFramebufferAttachmentParameteriv (GLenum target, GLenum attachment, GLenum
// pname, GLint* params);
Napi::Value WebGL2RenderingContext::GetFramebufferAttachmentParameter(
  Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint params{};
  GL_EXPORT::glGetFramebufferAttachmentParameteriv(args[0], args[1], args[2], &params);
  return CPPToNapi(info)(params);
}

// GL_EXPORT GLboolean glIsFramebuffer (GLuint framebuffer);
Napi::Value WebGL2RenderingContext::IsFramebuffer(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  return CPPToNapi(info)(GL_EXPORT::glIsFramebuffer(args[0]));
}

// GL_EXPORT void glBlitFramebuffer (GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint
// dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter);
Napi::Value WebGL2RenderingContext::BlitFramebuffer(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glBlitFramebuffer(
    args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9]);
  return info.Env().Undefined();
}

// GL_EXPORT void glFramebufferTextureLayer (GLenum target,GLenum attachment, GLuint texture,GLint
// level,GLint layer);
Napi::Value WebGL2RenderingContext::FramebufferTextureLayer(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glFramebufferTextureLayer(args[0], args[1], args[2], args[3], args[4]);
  return info.Env().Undefined();
}

// GL_EXPORT void glInvalidateFramebuffer (GLenum target, GLsizei numAttachments, const GLenum*
// attachments);
Napi::Value WebGL2RenderingContext::InvalidateFramebuffer(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glInvalidateFramebuffer(args[0], args[1], args[2]);
  return info.Env().Undefined();
}

// GL_EXPORT void glInvalidateSubFramebuffer (GLenum target, GLsizei numAttachments, const GLenum*
// attachments, GLint x, GLint y, GLsizei width, GLsizei height);
Napi::Value WebGL2RenderingContext::InvalidateSubFramebuffer(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glInvalidateSubFramebuffer(
    args[0], args[1], args[2], args[3], args[4], args[5], args[6]);
  return info.Env().Undefined();
}

}  // namespace nv
