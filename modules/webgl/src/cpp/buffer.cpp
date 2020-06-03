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

// GL_EXPORT void glBindBuffer (GLenum target, GLuint buffer);
Napi::Value WebGL2RenderingContext::BindBuffer(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glBindBuffer(FromJS(info[0]), FromJS(info[1]));
  return env.Undefined();
}

// GL_EXPORT void glBufferData (GLenum target, GLsizeiptr size, const void* data, GLenum usage);
Napi::Value WebGL2RenderingContext::BufferData(Napi::CallbackInfo const& info) {
  auto env      = info.Env();
  GLuint target = FromJS(info[0]);
  GLuint usage  = FromJS(info[2]);
  if (info[1].IsNull() || info[1].IsEmpty() || info[1].IsNumber()) {
    GL_EXPORT::glBufferData(target, FromJS(info[1]), NULL, usage);
  } else {
    std::pair<size_t, uint8_t*> ptr = FromJS(info[1]);
    GL_EXPORT::glBufferData(target, ptr.first, ptr.second, usage);
  }
  return env.Undefined();
}

// GL_EXPORT void glBufferSubData (GLenum target, GLintptr offset, GLsizeiptr size, const void*
// data);
Napi::Value WebGL2RenderingContext::BufferSubData(Napi::CallbackInfo const& info) {
  auto env                        = info.Env();
  std::pair<size_t, uint8_t*> ptr = FromJS(info[2]);
  if (ptr.first > 0 && ptr.second != nullptr) {
    GL_EXPORT::glBufferSubData(FromJS(info[0]), FromJS(info[1]), ptr.first, ptr.second);
  }
  return env.Undefined();
}

// GL_EXPORT void glCreateBuffers (GLsizei n, GLuint* buffers);
Napi::Value WebGL2RenderingContext::CreateBuffer(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GLuint buffer{};
  GL_EXPORT::glCreateBuffers(1, &buffer);
  return WebGLBuffer::New(buffer);
}

// GL_EXPORT void glCreateBuffers (GLsizei n, GLuint* buffers);
Napi::Value WebGL2RenderingContext::CreateBuffers(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  std::vector<GLuint> buffers(FromJS(info[0]).operator size_t());
  GL_EXPORT::glCreateBuffers(buffers.size(), buffers.data());
  return ToNapi(env)(buffers);
}

// GL_EXPORT void glDeleteBuffers (GLsizei n, const GLuint* buffers);
Napi::Value WebGL2RenderingContext::DeleteBuffer(Napi::CallbackInfo const& info) {
  auto env            = info.Env();
  const GLuint buffer = FromJS(info[0]);
  GL_EXPORT::glDeleteBuffers(1, &buffer);
  return env.Undefined();
}

// GL_EXPORT void glDeleteBuffers (GLsizei n, const GLuint* buffers);
Napi::Value WebGL2RenderingContext::DeleteBuffers(Napi::CallbackInfo const& info) {
  auto env                    = info.Env();
  std::vector<GLuint> buffers = FromJS(info[0]);
  GL_EXPORT::glDeleteBuffers(buffers.size(), buffers.data());
  return env.Undefined();
}

// GL_EXPORT void glGetBufferParameteriv (GLenum target, GLenum pname, GLint* params);
Napi::Value WebGL2RenderingContext::GetBufferParameter(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GLint params{};
  GL_EXPORT::glGetBufferParameteriv(FromJS(info[0]), FromJS(info[1]), &params);
  return ToNapi(env)(params);
}

// GL_EXPORT GLboolean glIsBuffer (GLuint buffer);
Napi::Value WebGL2RenderingContext::IsBuffer(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  return ToNapi(env)(GL_EXPORT::glIsBuffer(FromJS(info[0])));
}

// GL_EXPORT void glBindBufferBase (GLenum target, GLuint index, GLuint buffer);
Napi::Value WebGL2RenderingContext::BindBufferBase(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glBindBufferBase(FromJS(info[0]), FromJS(info[1]), FromJS(info[2]));
  return env.Undefined();
}

// GL_EXPORT void glBindBufferRange (GLenum target, GLuint index, GLuint buffer, GLintptr offset,
// GLsizeiptr size);
Napi::Value WebGL2RenderingContext::BindBufferRange(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glBindBufferRange(
    FromJS(info[0]), FromJS(info[1]), FromJS(info[2]), FromJS(info[3]), FromJS(info[4]));
  return env.Undefined();
}

// GL_EXPORT void glClearBufferfv (GLenum buffer, GLint drawBuffer, const GLfloat* value);
Napi::Value WebGL2RenderingContext::ClearBufferfv(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glClearBufferfv(FromJS(info[0]), FromJS(info[1]), FromJS(info[2]));
  return env.Undefined();
}

// GL_EXPORT void glClearBufferiv (GLenum buffer, GLint drawBuffer, const GLint* value);
Napi::Value WebGL2RenderingContext::ClearBufferiv(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glClearBufferiv(FromJS(info[0]), FromJS(info[1]), FromJS(info[2]));
  return env.Undefined();
}

// GL_EXPORT void glClearBufferfi (GLenum buffer, GLint drawBuffer, GLfloat depth, GLint stencil);
Napi::Value WebGL2RenderingContext::ClearBufferfi(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glClearBufferfi(FromJS(info[0]), FromJS(info[1]), FromJS(info[2]), FromJS(info[3]));
  return env.Undefined();
}

// GL_EXPORT void glClearBufferuiv (GLenum buffer, GLint drawBuffer, const GLuint* value);
Napi::Value WebGL2RenderingContext::ClearBufferuiv(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glClearBufferuiv(FromJS(info[0]), FromJS(info[1]), FromJS(info[2]));
  return env.Undefined();
}

// GL_EXPORT void glCopyBufferSubData (GLenum readtarget, GLenum writetarget, GLintptr readoffset,
// GLintptr writeoffset, GLsizeiptr size);
Napi::Value WebGL2RenderingContext::CopyBufferSubData(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glCopyBufferSubData(
    FromJS(info[0]), FromJS(info[1]), FromJS(info[2]), FromJS(info[3]), FromJS(info[4]));
  return env.Undefined();
}

// GL_EXPORT void glDrawBuffers (GLsizei n, const GLenum* bufs);
Napi::Value WebGL2RenderingContext::DrawBuffers(Napi::CallbackInfo const& info) {
  auto env                    = info.Env();
  std::vector<GLuint> buffers = FromJS(info[0]);
  GL_EXPORT::glDrawBuffers(buffers.size(), buffers.data());
  return env.Undefined();
}

// GL_EXPORT void glGetBufferSubData (GLenum target, GLintptr offset, GLsizeiptr size, void* data);
Napi::Value WebGL2RenderingContext::GetBufferSubData(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glGetBufferSubData(FromJS(info[0]), FromJS(info[1]), FromJS(info[2]), FromJS(info[3]));
  return env.Undefined();
}

// GL_EXPORT void glReadBuffer (GLenum mode);
Napi::Value WebGL2RenderingContext::ReadBuffer(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glReadBuffer(FromJS(info[0]));
  return env.Undefined();
}

}  // namespace node_webgl
