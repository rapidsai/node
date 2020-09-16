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

// GLEWAPI void glBindAttribLocation (GLuint program, GLuint index, const GLchar* name);
Napi::Value WebGL2RenderingContext::BindAttribLocation(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  std::string name  = args[2];
  GL_EXPORT::glBindAttribLocation(args[0], args[1], name.data());
  return info.Env().Undefined();
}

// GLEWAPI void glDisableVertexAttribArray (GLuint index);
Napi::Value WebGL2RenderingContext::DisableVertexAttribArray(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glDisableVertexAttribArray(args[0]);
  return info.Env().Undefined();
}

// GLEWAPI void glEnableVertexAttribArray (GLuint index);
Napi::Value WebGL2RenderingContext::EnableVertexAttribArray(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glEnableVertexAttribArray(args[0]);
  return info.Env().Undefined();
}

// GLEWAPI void glGetActiveAttrib (GLuint program, GLuint index, GLsizei maxLength, GLsizei* length,
// GLint* size, GLenum* type, GLchar* name);
Napi::Value WebGL2RenderingContext::GetActiveAttrib(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLuint program    = args[0];
  GLuint attrib_    = args[1];
  GLint max_len{0};
  GL_EXPORT::glGetProgramiv(program, GL_ACTIVE_ATTRIBUTE_MAX_LENGTH, &max_len);
  if (max_len > 0) {
    GLuint type{};
    GLint size{0}, length{0};
    GLchar* name = reinterpret_cast<GLchar*>(std::malloc(max_len));
    GL_EXPORT::glGetActiveAttrib(program, attrib_, max_len, &length, &size, &type, name);
    return WebGLActiveInfo::New(size, type, std::string{name, static_cast<size_t>(length)});
  }
  return info.Env().Null();
}

// GLEWAPI GLint glGetAttribLocation (GLuint program, const GLchar* name);
Napi::Value WebGL2RenderingContext::GetAttribLocation(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  std::string name  = args[1];
  return CPPToNapi(info)(GL_EXPORT::glGetAttribLocation(args[0], name.data()));
}

// GLEWAPI void glGetVertexAttribiv (GLuint index, GLenum pname, GLint* params);
// GLEWAPI void glGetVertexAttribfv (GLuint index, GLenum pname, GLfloat* params);
// GLEWAPI void glGetVertexAttribdv (GLuint index, GLenum pname, GLfloat* params);
Napi::Value WebGL2RenderingContext::GetVertexAttrib(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLuint index      = args[0];
  GLint pname       = args[1];
  switch (pname) {
    case GL_VERTEX_ATTRIB_ARRAY_ENABLED:
    case GL_VERTEX_ATTRIB_ARRAY_INTEGER:
    case GL_VERTEX_ATTRIB_ARRAY_NORMALIZED: {
      GLint value{};
      GL_EXPORT::glGetVertexAttribiv(index, pname, &value);
      return CPPToNapi(info)(static_cast<bool>(value));
    }
    case GL_VERTEX_ATTRIB_ARRAY_SIZE:
    case GL_VERTEX_ATTRIB_ARRAY_TYPE:
    case GL_VERTEX_ATTRIB_ARRAY_STRIDE:
    case GL_VERTEX_ATTRIB_ARRAY_DIVISOR:
    case GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING: {
      GLint value{};
      GL_EXPORT::glGetVertexAttribiv(index, pname, &value);
      return CPPToNapi(info)(value);
    }
    case GL_CURRENT_VERTEX_ATTRIB: {
      auto env = info.Env();
      auto buf = Napi::ArrayBuffer::New(env, sizeof(GLfloat) * 4);
      auto ptr = reinterpret_cast<GLfloat*>(buf.Data());
      GL_EXPORT::glGetVertexAttribfv(index, pname, ptr);
      return Napi::Float32Array::New(env, 4, buf, 0);
    }
  }
  GLEW_THROW(info.Env(), GL_INVALID_ENUM);
}

// GLEWAPI void glGetVertexAttribPointerv (GLuint index, GLenum pname, void** pointer);
Napi::Value WebGL2RenderingContext::GetVertexAttribPointerv(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  void* vertex_attrib{nullptr};
  GL_EXPORT::glGetVertexAttribPointerv(args[0], args[1], &vertex_attrib);
  return CPPToNapi(info)(vertex_attrib);
}

// GLEWAPI void glVertexAttrib1f (GLuint index, GLfloat x);
Napi::Value WebGL2RenderingContext::VertexAttrib1f(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glVertexAttrib1f(args[0], args[1]);
  return info.Env().Undefined();
}

// GLEWAPI void glVertexAttrib1fv (GLuint index, const GLfloat* v);
Napi::Value WebGL2RenderingContext::VertexAttrib1fv(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glVertexAttrib1fv(args[0], args[1]);
  return info.Env().Undefined();
}

// GLEWAPI void glVertexAttrib2f (GLuint index, GLfloat x, GLfloat y);
Napi::Value WebGL2RenderingContext::VertexAttrib2f(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glVertexAttrib2f(args[0], args[1], args[2]);
  return info.Env().Undefined();
}

// GLEWAPI void glVertexAttrib2fv (GLuint index, const GLfloat* v);
Napi::Value WebGL2RenderingContext::VertexAttrib2fv(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glVertexAttrib2fv(args[0], args[1]);
  return info.Env().Undefined();
}

// GLEWAPI void glVertexAttrib3f (GLuint index, GLfloat x, GLfloat y, GLfloat z);
Napi::Value WebGL2RenderingContext::VertexAttrib3f(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glVertexAttrib3f(args[0], args[1], args[2], args[3]);
  return info.Env().Undefined();
}

// GLEWAPI void glVertexAttrib3fv (GLuint index, const GLfloat* v);
Napi::Value WebGL2RenderingContext::VertexAttrib3fv(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glVertexAttrib3fv(args[0], args[1]);
  return info.Env().Undefined();
}

// GLEWAPI void glVertexAttrib4f (GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
Napi::Value WebGL2RenderingContext::VertexAttrib4f(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glVertexAttrib4f(args[0], args[1], args[2], args[3], args[4]);
  return info.Env().Undefined();
}

// GLEWAPI void glVertexAttrib4fv (GLuint index, const GLfloat* v);
Napi::Value WebGL2RenderingContext::VertexAttrib4fv(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glVertexAttrib4fv(args[0], args[1]);
  return info.Env().Undefined();
}

// GLEWAPI void glVertexAttribPointer (GLuint index, GLint size, GLenum type, GLboolean normalized,
// GLsizei stride, const void* pointer);
Napi::Value WebGL2RenderingContext::VertexAttribPointer(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glVertexAttribPointer(args[0], args[1], args[2], args[3], args[4], args[5]);
  return info.Env().Undefined();
}

// GLEWAPI void glVertexAttribI4i (GLuint index, GLint v0, GLint v1, GLint v2, GLint v3);
Napi::Value WebGL2RenderingContext::VertexAttribI4i(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glVertexAttribI4i(args[0], args[1], args[2], args[3], args[4]);
  return info.Env().Undefined();
}

// GLEWAPI void glVertexAttribI4iv (GLuint index, const GLint* v0);
Napi::Value WebGL2RenderingContext::VertexAttribI4iv(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glVertexAttribI4iv(args[0], args[1]);
  return info.Env().Undefined();
}

// GLEWAPI void glVertexAttribI4ui (GLuint index, GLuint v0, GLuint v1, GLuint v2, GLuint v3);
Napi::Value WebGL2RenderingContext::VertexAttribI4ui(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glVertexAttribI4ui(args[0], args[1], args[2], args[3], args[4]);
  return info.Env().Undefined();
}

// GLEWAPI void glVertexAttribI4uiv (GLuint index, const GLuint* v0);
Napi::Value WebGL2RenderingContext::VertexAttribI4uiv(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glVertexAttribI4uiv(args[0], args[1]);
  return info.Env().Undefined();
}

// GLEWAPI void glVertexAttribIPointer (GLuint index, GLint size, GLenum type, GLsizei stride, const
// void*pointer);
Napi::Value WebGL2RenderingContext::VertexAttribIPointer(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glVertexAttribIPointer(args[0], args[1], args[2], args[3], args[4]);
  return info.Env().Undefined();
}

}  // namespace nv
