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

// GL_EXPORT void glBindBuffer (GLenum target, GLuint buffer);
void WebGL2RenderingContext::BindBuffer(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glBindBuffer(args[0], args[1]);
}

// GL_EXPORT void glBufferData (GLenum target, GLsizeiptr size, const void* data, GLenum usage);
void WebGL2RenderingContext::BufferData(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLuint target     = args[0];
  GLuint usage      = args[2];
  if (info[1].IsNull() || info[1].IsEmpty() || info[1].IsNumber()) {
    GL_EXPORT::glBufferData(target, args[1], NULL, usage);
  } else {
    Span<char> ptr = args[1];
    GL_EXPORT::glBufferData(target, ptr.size(), ptr.data(), usage);
  }
}

// GL_EXPORT void glBufferSubData (GLenum target, GLintptr offset, GLsizeiptr size, const void*
// data);
void WebGL2RenderingContext::BufferSubData(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  Span<char> ptr    = args[2];
  if (ptr.size() > 0 && ptr.data() != nullptr) {
    GL_EXPORT::glBufferSubData(args[0], args[1], ptr.size(), ptr.data());
  }
}

// GL_EXPORT void glCreateBuffers (GLsizei n, GLuint* buffers);
Napi::Value WebGL2RenderingContext::CreateBuffer(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLuint buffer{};
  GL_EXPORT::glCreateBuffers(1, &buffer);
  return WebGLBuffer::New(info.Env(), buffer);
}

// GL_EXPORT void glCreateBuffers (GLsizei n, GLuint* buffers);
Napi::Value WebGL2RenderingContext::CreateBuffers(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  std::vector<GLuint> buffers(args[0].operator size_t());
  GL_EXPORT::glCreateBuffers(buffers.size(), buffers.data());
  return CPPToNapi(info)(buffers);
}

// GL_EXPORT void glDeleteBuffers (GLsizei n, const GLuint* buffers);
void WebGL2RenderingContext::DeleteBuffer(Napi::CallbackInfo const& info) {
  CallbackArgs args   = info;
  const GLuint buffer = args[0];
  GL_EXPORT::glDeleteBuffers(1, &buffer);
}

// GL_EXPORT void glDeleteBuffers (GLsizei n, const GLuint* buffers);
void WebGL2RenderingContext::DeleteBuffers(Napi::CallbackInfo const& info) {
  CallbackArgs args           = info;
  std::vector<GLuint> buffers = args[0];
  GL_EXPORT::glDeleteBuffers(buffers.size(), buffers.data());
}

// GL_EXPORT void glGetBufferParameteriv (GLenum target, GLenum pname, GLint* params);
Napi::Value WebGL2RenderingContext::GetBufferParameter(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint params{};
  GL_EXPORT::glGetBufferParameteriv(args[0], args[1], &params);
  return CPPToNapi(info)(params);
}

// GL_EXPORT GLboolean glIsBuffer (GLuint buffer);
Napi::Value WebGL2RenderingContext::IsBuffer(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  return CPPToNapi(info)(GL_EXPORT::glIsBuffer(args[0]));
}

// GL_EXPORT void glBindBufferBase (GLenum target, GLuint index, GLuint buffer);
void WebGL2RenderingContext::BindBufferBase(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glBindBufferBase(args[0], args[1], args[2]);
}

// GL_EXPORT void glBindBufferRange (GLenum target, GLuint index, GLuint buffer, GLintptr offset,
// GLsizeiptr size);
void WebGL2RenderingContext::BindBufferRange(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glBindBufferRange(args[0], args[1], args[2], args[3], args[4]);
}

// GL_EXPORT void glClearBufferfv (GLenum buffer, GLint drawBuffer, const GLfloat* value);
void WebGL2RenderingContext::ClearBufferfv(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glClearBufferfv(args[0], args[1], args[2]);
}

// GL_EXPORT void glClearBufferiv (GLenum buffer, GLint drawBuffer, const GLint* value);
void WebGL2RenderingContext::ClearBufferiv(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glClearBufferiv(args[0], args[1], args[2]);
}

// GL_EXPORT void glClearBufferfi (GLenum buffer, GLint drawBuffer, GLfloat depth, GLint stencil);
void WebGL2RenderingContext::ClearBufferfi(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glClearBufferfi(args[0], args[1], args[2], args[3]);
}

// GL_EXPORT void glClearBufferuiv (GLenum buffer, GLint drawBuffer, const GLuint* value);
void WebGL2RenderingContext::ClearBufferuiv(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glClearBufferuiv(args[0], args[1], args[2]);
}

// GL_EXPORT void glCopyBufferSubData (GLenum readtarget, GLenum writetarget, GLintptr readoffset,
// GLintptr writeoffset, GLsizeiptr size);
void WebGL2RenderingContext::CopyBufferSubData(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glCopyBufferSubData(args[0], args[1], args[2], args[3], args[4]);
}

// GL_EXPORT void glDrawBuffers (GLsizei n, const GLenum* bufs);
void WebGL2RenderingContext::DrawBuffers(Napi::CallbackInfo const& info) {
  CallbackArgs args           = info;
  std::vector<GLuint> buffers = args[0];
  GL_EXPORT::glDrawBuffers(buffers.size(), buffers.data());
}

// GL_EXPORT void glGetBufferSubData (GLenum target, GLintptr offset, GLsizeiptr size, void* data);
void WebGL2RenderingContext::GetBufferSubData(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glGetBufferSubData(args[0], args[1], args[2], args[3]);
}

// GL_EXPORT void glReadBuffer (GLenum mode);
void WebGL2RenderingContext::ReadBuffer(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glReadBuffer(args[0]);
}

}  // namespace nv
