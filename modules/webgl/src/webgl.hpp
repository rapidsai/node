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

#pragma once

#include "gl.hpp"

#include <nv_node/objectwrap.hpp>

#include <napi.h>

#ifndef GL_GPU_DISJOINT
#define GL_GPU_DISJOINT 0x8FBB
#endif

#ifndef GL_UNMASKED_VENDOR_WEBGL
#define GL_UNMASKED_VENDOR_WEBGL 0x9245
#endif

#ifndef GL_UNMASKED_RENDERER_WEBGL
#define GL_UNMASKED_RENDERER_WEBGL 0x9246
#endif

#ifndef GL_UNPACK_FLIP_Y_WEBGL
#define GL_UNPACK_FLIP_Y_WEBGL 0x9240
#endif

#ifndef GL_UNPACK_PREMULTIPLY_ALPHA_WEBGL
#define GL_UNPACK_PREMULTIPLY_ALPHA_WEBGL 0x9241
#endif

#ifndef GL_CONTEXT_LOST_WEBGL
#define GL_CONTEXT_LOST_WEBGL 0x9242
#endif

#ifndef GL_UNPACK_COLORSPACE_CONVERSION_WEBGL
#define GL_UNPACK_COLORSPACE_CONVERSION_WEBGL 0x9243
#endif

#ifndef GL_BROWSER_DEFAULT_WEBGL
#define GL_BROWSER_DEFAULT_WEBGL 0x9244
#endif

#ifndef GL_MAX_CLIENT_WAIT_TIMEOUT_WEBGL
#define GL_MAX_CLIENT_WAIT_TIMEOUT_WEBGL 0x9247
#endif

namespace nv {

struct WebGLActiveInfo : public EnvLocalObjectWrap<WebGLActiveInfo> {
  // using EnvLocalObjectWrap<WebGLActiveInfo>::New;

  static Napi::Function Init(Napi::Env const& env, Napi::Object exports);
  static wrapper_t New(Napi::Env const& env, GLint size, GLuint type, std::string name);

  WebGLActiveInfo(Napi::CallbackInfo const& info);

 private:
  Napi::Value ToString(Napi::CallbackInfo const& info);
  Napi::Value GetSize(Napi::CallbackInfo const& info);
  Napi::Value GetType(Napi::CallbackInfo const& info);
  Napi::Value GetName(Napi::CallbackInfo const& info);

  GLint size_{0};
  GLuint type_{0};
  std::string name_{""};
};

struct WebGLShaderPrecisionFormat : public EnvLocalObjectWrap<WebGLShaderPrecisionFormat> {
  // using EnvLocalObjectWrap<WebGLShaderPrecisionFormat>::New;

  static Napi::Function Init(Napi::Env const& env, Napi::Object exports);
  static wrapper_t New(Napi::Env const& env, GLint rangeMax, GLint rangeMin, GLint precision);

  WebGLShaderPrecisionFormat(Napi::CallbackInfo const& info);

 private:
  Napi::Value ToString(Napi::CallbackInfo const& info);
  Napi::Value GetRangeMin(Napi::CallbackInfo const& info);
  Napi::Value GetRangeMax(Napi::CallbackInfo const& info);
  Napi::Value GetPrecision(Napi::CallbackInfo const& info);

  GLint rangeMin_{0};
  GLint rangeMax_{0};
  GLint precision_{0};
};

struct WebGLBuffer : public EnvLocalObjectWrap<WebGLBuffer> {
  // using EnvLocalObjectWrap<WebGLBuffer>::New;

  static Napi::Function Init(Napi::Env const& env, Napi::Object exports);
  static wrapper_t New(Napi::Env const& env, GLuint value);

  WebGLBuffer(Napi::CallbackInfo const& info);
  operator GLuint() { return this->value_; }

 private:
  Napi::Value ToString(Napi::CallbackInfo const& info);
  Napi::Value GetValue(Napi::CallbackInfo const& info);
  GLuint value_{0};
};

struct WebGLContextEvent : public EnvLocalObjectWrap<WebGLContextEvent> {
  // using EnvLocalObjectWrap<WebGLContextEvent>::New;

  static Napi::Function Init(Napi::Env const& env, Napi::Object exports);
  static wrapper_t New(Napi::Env const& env, GLuint value);

  WebGLContextEvent(Napi::CallbackInfo const& info);
  operator GLuint() { return this->value_; }

 private:
  Napi::Value ToString(Napi::CallbackInfo const& info);
  Napi::Value GetValue(Napi::CallbackInfo const& info);
  GLuint value_{0};
};

struct WebGLFramebuffer : public EnvLocalObjectWrap<WebGLFramebuffer> {
  // using EnvLocalObjectWrap<WebGLFramebuffer>::New;

  static Napi::Function Init(Napi::Env const& env, Napi::Object exports);
  static wrapper_t New(Napi::Env const& env, GLuint value);

  WebGLFramebuffer(Napi::CallbackInfo const& info);
  operator GLuint() { return this->value_; }

 private:
  Napi::Value ToString(Napi::CallbackInfo const& info);
  Napi::Value GetValue(Napi::CallbackInfo const& info);
  GLuint value_{0};
};

struct WebGLProgram : public EnvLocalObjectWrap<WebGLProgram> {
  // using EnvLocalObjectWrap<WebGLProgram>::New;

  static Napi::Function Init(Napi::Env const& env, Napi::Object exports);
  static wrapper_t New(Napi::Env const& env, GLuint value);

  WebGLProgram(Napi::CallbackInfo const& info);
  operator GLuint() { return this->value_; }

 private:
  Napi::Value ToString(Napi::CallbackInfo const& info);
  Napi::Value GetValue(Napi::CallbackInfo const& info);
  GLuint value_{0};
};

struct WebGLQuery : public EnvLocalObjectWrap<WebGLQuery> {
  // using EnvLocalObjectWrap<WebGLQuery>::New;

  static Napi::Function Init(Napi::Env const& env, Napi::Object exports);
  static wrapper_t New(Napi::Env const& env, GLuint value);

  WebGLQuery(Napi::CallbackInfo const& info);
  operator GLuint() { return this->value_; }

 private:
  Napi::Value ToString(Napi::CallbackInfo const& info);
  Napi::Value GetValue(Napi::CallbackInfo const& info);
  GLuint value_{0};
};

struct WebGLRenderbuffer : public EnvLocalObjectWrap<WebGLRenderbuffer> {
  // using EnvLocalObjectWrap<WebGLRenderbuffer>::New;

  static Napi::Function Init(Napi::Env const& env, Napi::Object exports);
  static wrapper_t New(Napi::Env const& env, GLuint value);

  WebGLRenderbuffer(Napi::CallbackInfo const& info);
  operator GLuint() { return this->value_; }

 private:
  Napi::Value ToString(Napi::CallbackInfo const& info);
  Napi::Value GetValue(Napi::CallbackInfo const& info);
  GLuint value_{0};
};

struct WebGLSampler : public EnvLocalObjectWrap<WebGLSampler> {
  // using EnvLocalObjectWrap<WebGLSampler>::New;

  static Napi::Function Init(Napi::Env const& env, Napi::Object exports);
  static wrapper_t New(Napi::Env const& env, GLuint value);

  WebGLSampler(Napi::CallbackInfo const& info);
  operator GLuint() { return this->value_; }

 private:
  Napi::Value ToString(Napi::CallbackInfo const& info);
  Napi::Value GetValue(Napi::CallbackInfo const& info);
  GLuint value_{0};
};

struct WebGLShader : public EnvLocalObjectWrap<WebGLShader> {
  // using EnvLocalObjectWrap<WebGLShader>::New;

  static Napi::Function Init(Napi::Env const& env, Napi::Object exports);
  static wrapper_t New(Napi::Env const& env, GLuint value);

  WebGLShader(Napi::CallbackInfo const& info);
  operator GLuint() { return this->value_; }

 private:
  Napi::Value ToString(Napi::CallbackInfo const& info);
  Napi::Value GetValue(Napi::CallbackInfo const& info);
  GLuint value_{0};
};

struct WebGLSync : public EnvLocalObjectWrap<WebGLSync> {
  // using EnvLocalObjectWrap<WebGLSync>::New;

  static Napi::Function Init(Napi::Env const& env, Napi::Object exports);
  static wrapper_t New(Napi::Env const& env, GLsync value);

  WebGLSync(Napi::CallbackInfo const& info);
  operator GLsync() { return this->value_; }

 private:
  Napi::Value ToString(Napi::CallbackInfo const& info);
  Napi::Value GetValue(Napi::CallbackInfo const& info);
  GLsync value_{0};
};

struct WebGLTexture : public EnvLocalObjectWrap<WebGLTexture> {
  // using EnvLocalObjectWrap<WebGLTexture>::New;

  static Napi::Function Init(Napi::Env const& env, Napi::Object exports);
  static wrapper_t New(Napi::Env const& env, GLuint value);

  WebGLTexture(Napi::CallbackInfo const& info);
  operator GLuint() { return this->value_; }

 private:
  Napi::Value ToString(Napi::CallbackInfo const& info);
  Napi::Value GetValue(Napi::CallbackInfo const& info);
  GLuint value_{0};
};

struct WebGLTransformFeedback : public EnvLocalObjectWrap<WebGLTransformFeedback> {
  // using EnvLocalObjectWrap<WebGLTransformFeedback>::New;

  static Napi::Function Init(Napi::Env const& env, Napi::Object exports);
  static wrapper_t New(Napi::Env const& env, GLuint value);

  WebGLTransformFeedback(Napi::CallbackInfo const& info);
  operator GLuint() { return this->value_; }

 private:
  Napi::Value ToString(Napi::CallbackInfo const& info);
  Napi::Value GetValue(Napi::CallbackInfo const& info);
  GLuint value_{0};
};

struct WebGLUniformLocation : public EnvLocalObjectWrap<WebGLUniformLocation> {
  // using EnvLocalObjectWrap<WebGLUniformLocation>::New;

  static Napi::Function Init(Napi::Env const& env, Napi::Object exports);
  static wrapper_t New(Napi::Env const& env, GLint value);

  WebGLUniformLocation(Napi::CallbackInfo const& info);
  operator GLint() { return this->value_; }

 private:
  Napi::Value ToString(Napi::CallbackInfo const& info);
  Napi::Value GetValue(Napi::CallbackInfo const& info);
  GLint value_{0};
};

struct WebGLVertexArrayObject : public EnvLocalObjectWrap<WebGLVertexArrayObject> {
  // using EnvLocalObjectWrap<WebGLVertexArrayObject>::New;

  static Napi::Function Init(Napi::Env const& env, Napi::Object exports);
  static wrapper_t New(Napi::Env const& env, GLuint value);

  WebGLVertexArrayObject(Napi::CallbackInfo const& info);
  operator GLuint() { return this->value_; }

 private:
  Napi::Value ToString(Napi::CallbackInfo const& info);
  Napi::Value GetValue(Napi::CallbackInfo const& info);
  GLuint value_{0};
};

struct WebGL2RenderingContext : public EnvLocalObjectWrap<WebGL2RenderingContext> {
  // using EnvLocalObjectWrap<WebGL2RenderingContext>::New;

  static Napi::Function Init(Napi::Env const& env, Napi::Object exports);

  WebGL2RenderingContext(Napi::CallbackInfo const& info);

 private:
  ///
  // misc
  ///
  // GL_EXPORT void glClear (GLbitfield mask);
  void Clear(Napi::CallbackInfo const& info);
  // GL_EXPORT void glClearColor (GLclampf red, GLclampf green, GLclampf blue, GLclampf alpha);
  void ClearColor(Napi::CallbackInfo const& info);
  // GL_EXPORT void glClearDepth (GLclampd depth);
  void ClearDepth(Napi::CallbackInfo const& info);
  // GL_EXPORT void glColorMask (GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha);
  void ColorMask(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCullFace (GLenum mode);
  void CullFace(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDepthFunc (GLenum func);
  void DepthFunc(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDepthMask (GLboolean flag);
  void DepthMask(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDepthRange (GLclampd zNear, GLclampd zFar);
  void DepthRange(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDisable (GLenum cap);
  void Disable(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDrawArrays (GLenum mode, GLint first, GLsizei count);
  void DrawArrays(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDrawElements (GLenum mode, GLsizei count, GLenum type, const void *indices);
  void DrawElements(Napi::CallbackInfo const& info);
  // GL_EXPORT void glEnable (GLenum cap);
  void Enable(Napi::CallbackInfo const& info);
  // GL_EXPORT void glFinish (void);
  void Finish(Napi::CallbackInfo const& info);
  // GL_EXPORT void glFlush (void);
  void Flush(Napi::CallbackInfo const& info);
  // GL_EXPORT void glFrontFace (GLenum mode);
  void FrontFace(Napi::CallbackInfo const& info);
  // GL_EXPORT GLenum glGetError (void);
  Napi::Value GetError(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetParameter (GLint pname);
  Napi::Value GetParameter(Napi::CallbackInfo const& info);
  // GL_EXPORT const GLubyte * glGetString (GL_EXTENSIONS);
  Napi::Value GetSupportedExtensions(Napi::CallbackInfo const& info);
  // GL_EXPORT void glHint (GLenum target, GLenum mode);
  void Hint(Napi::CallbackInfo const& info);
  // GL_EXPORT GLboolean glIsEnabled (GLenum cap);
  Napi::Value IsEnabled(Napi::CallbackInfo const& info);
  // GL_EXPORT void glLineWidth (GLfloat width);
  void LineWidth(Napi::CallbackInfo const& info);
  // GL_EXPORT void glPixelStorei (GLenum pname, GLint param);
  void PixelStorei(Napi::CallbackInfo const& info);
  // GL_EXPORT void glPolygonOffset (GLfloat factor, GLfloat units);
  void PolygonOffset(Napi::CallbackInfo const& info);
  // GL_EXPORT void glReadPixels (GLint x, GLint y, GLsizei width, GLsizei height, GLenum format,
  // GLenum type, void *pixels);
  void ReadPixels(Napi::CallbackInfo const& info);
  // GL_EXPORT void glScissor (GLint x, GLint y, GLsizei width, GLsizei height);
  void Scissor(Napi::CallbackInfo const& info);
  // GL_EXPORT void glViewport (GLint x, GLint y, GLsizei width, GLsizei height);
  void Viewport(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDrawRangeElements (GLenum mode, GLuint start, GLuint end, GLsizei count,
  // GLenum type, const void *indices);
  void DrawRangeElements(Napi::CallbackInfo const& info);
  // GL_EXPORT void glSampleCoverage (GLclampf value, GLboolean invert);
  void SampleCoverage(Napi::CallbackInfo const& info);
  // GL_EXPORT GLint glGetFragDataLocation (GLuint program, const GLchar* name);
  Napi::Value GetFragDataLocation(Napi::CallbackInfo const& info);

  ///
  // attrib
  ///

  // GL_EXPORT void glBindAttribLocation (GLuint program, GLuint index, const GLchar* name);
  void BindAttribLocation(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDisableVertexAttribArray (GLuint index);
  void DisableVertexAttribArray(Napi::CallbackInfo const& info);
  // GL_EXPORT void glEnableVertexAttribArray (GLuint index);
  void EnableVertexAttribArray(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetActiveAttrib (GLuint program, GLuint index, GLsizei maxLength, GLsizei*
  // length, GLint* size, GLenum* type, GLchar* name);
  Napi::Value GetActiveAttrib(Napi::CallbackInfo const& info);
  // GL_EXPORT GLint glGetAttribLocation (GLuint program, const GLchar* name);
  Napi::Value GetAttribLocation(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetVertexAttribdv (GLuint index, GLenum pname, GLdouble* params);
  // GL_EXPORT void glGetVertexAttribfv (GLuint index, GLenum pname, GLfloat* params);
  // GL_EXPORT void glGetVertexAttribiv (GLuint index, GLenum pname, GLint* params);
  Napi::Value GetVertexAttrib(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetVertexAttribPointerv (GLuint index, GLenum pname, void** pointer);
  Napi::Value GetVertexAttribPointerv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glVertexAttrib1f (GLuint index, GLfloat x);
  void VertexAttrib1f(Napi::CallbackInfo const& info);
  // GL_EXPORT void glVertexAttrib1fv (GLuint index, const GLfloat* v);
  void VertexAttrib1fv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glVertexAttrib2f (GLuint index, GLfloat x, GLfloat y);
  void VertexAttrib2f(Napi::CallbackInfo const& info);
  // GL_EXPORT void glVertexAttrib2fv (GLuint index, const GLfloat* v);
  void VertexAttrib2fv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glVertexAttrib3f (GLuint index, GLfloat x, GLfloat y, GLfloat z);
  void VertexAttrib3f(Napi::CallbackInfo const& info);
  // GL_EXPORT void glVertexAttrib3fv (GLuint index, const GLfloat* v);
  void VertexAttrib3fv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glVertexAttrib4f (GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
  void VertexAttrib4f(Napi::CallbackInfo const& info);
  // GL_EXPORT void glVertexAttrib4fv (GLuint index, const GLfloat* v);
  void VertexAttrib4fv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glVertexAttribPointer (GLuint index, GLint size, GLenum type, GLboolean
  // normalized, GLsizei stride, const void* pointer);
  void VertexAttribPointer(Napi::CallbackInfo const& info);
  // GL_EXPORT void glVertexAttribI4i (GLuint index, GLint v0, GLint v1, GLint v2, GLint v3);
  void VertexAttribI4i(Napi::CallbackInfo const& info);
  // GL_EXPORT void glVertexAttribI4iv (GLuint index, const GLint* v0);
  void VertexAttribI4iv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glVertexAttribI4ui (GLuint index, GLuint v0, GLuint v1, GLuint v2, GLuint v3);
  void VertexAttribI4ui(Napi::CallbackInfo const& info);
  // GL_EXPORT void glVertexAttribI4uiv (GLuint index, const GLuint* v0);
  void VertexAttribI4uiv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glVertexAttribIPointer (GLuint index, GLint size, GLenum type, GLsizei stride,
  // const void*pointer);
  void VertexAttribIPointer(Napi::CallbackInfo const& info);

  ///
  // blend
  ///
  // GL_EXPORT void glBlendColor (GLclampf red, GLclampf green, GLclampf blue, GLclampf alpha);
  void BlendColor(Napi::CallbackInfo const& info);
  // GL_EXPORT void glBlendEquation (GLenum mode);
  void BlendEquation(Napi::CallbackInfo const& info);
  // GL_EXPORT void glBlendEquationSeparate (GLenum modeRGB, GLenum modeAlpha);
  void BlendEquationSeparate(Napi::CallbackInfo const& info);
  // GL_EXPORT void glBlendFunc (GLenum sfactor, GLenum dfactor);
  void BlendFunc(Napi::CallbackInfo const& info);
  // GL_EXPORT void glBlendFuncSeparate (GLenum sfactorRGB, GLenum dfactorRGB, GLenum sfactorAlpha,
  // GLenum dfactorAlpha);
  void BlendFuncSeparate(Napi::CallbackInfo const& info);

  ///
  // buffer
  ///
  // GL_EXPORT void glBindBuffer (GLenum target, GLuint buffer);
  void BindBuffer(Napi::CallbackInfo const& info);
  // GL_EXPORT void glBufferData (GLenum target, GLsizeiptr size, const void* data, GLenum usage);
  void BufferData(Napi::CallbackInfo const& info);
  // GL_EXPORT void glBufferSubData (GLenum target, GLintptr offset, GLsizeiptr size, const void*
  // data);
  void BufferSubData(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCreateBuffers (GLsizei n, GLuint* buffers);
  Napi::Value CreateBuffer(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCreateBuffers (GLsizei n, GLuint* buffers);
  Napi::Value CreateBuffers(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteBuffers (GLsizei n, const GLuint* buffers);
  void DeleteBuffer(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteBuffers (GLsizei n, const GLuint* buffers);
  void DeleteBuffers(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetBufferParameter (GLenum target, GLenum pname, GLint* params);
  Napi::Value GetBufferParameter(Napi::CallbackInfo const& info);
  // GL_EXPORT GLboolean glIsBuffer (GLuint buffer);
  Napi::Value IsBuffer(Napi::CallbackInfo const& info);
  // GL_EXPORT void glBindBufferBase (GLenum target, GLuint index, GLuint buffer);
  void BindBufferBase(Napi::CallbackInfo const& info);
  // GL_EXPORT void glBindBufferRange (GLenum target, GLuint index, GLuint buffer, GLintptr offset,
  // GLsizeiptr size);
  void BindBufferRange(Napi::CallbackInfo const& info);
  // GL_EXPORT void glClearBufferfv (GLenum buffer, GLint drawBuffer, const GLfloat* value);
  void ClearBufferfv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glClearBufferiv (GLenum buffer, GLint drawBuffer, const GLint* value);
  void ClearBufferiv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glClearBufferuiv (GLenum buffer, GLint drawBuffer, const GLuint* value);
  void ClearBufferuiv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glClearBufferfi (GLenum buffer, GLint drawBuffer, GLfloat depth, GLint stencil);
  void ClearBufferfi(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCopyBufferSubData (GLenum readtarget, GLenum writetarget, GLintptr readoffset,
  // GLintptr writeoffset, GLsizeiptr size);
  void CopyBufferSubData(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDrawBuffers (GLsizei n, const GLenum* bufs);
  void DrawBuffers(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetBufferSubData (GLenum target, GLintptr offset, GLsizeiptr size, void*
  // data);
  void GetBufferSubData(Napi::CallbackInfo const& info);
  // GL_EXPORT void glReadBuffer (GLenum mode);
  void ReadBuffer(Napi::CallbackInfo const& info);

  ///
  // framebuffer
  ///
  // GL_EXPORT void glBindFramebuffer (GLenum target, GLuint framebuffer);
  void BindFramebuffer(Napi::CallbackInfo const& info);
  // GL_EXPORT GLenum glCheckFramebufferStatus (GLenum target);
  Napi::Value CheckFramebufferStatus(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCreateFramebuffers (GLsizei n, GLuint* framebuffers);
  Napi::Value CreateFramebuffer(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCreateFramebuffers (GLsizei n, GLuint* framebuffers);
  Napi::Value CreateFramebuffers(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteFramebuffers (GLsizei n, const GLuint* framebuffers);
  void DeleteFramebuffer(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteFramebuffers (GLsizei n, const GLuint* framebuffers);
  void DeleteFramebuffers(Napi::CallbackInfo const& info);
  // GL_EXPORT void glFramebufferRenderbuffer (GLenum target, GLenum attachment, GLenum
  // renderbuffertarget, GLuint renderbuffer);
  void FramebufferRenderbuffer(Napi::CallbackInfo const& info);
  // GL_EXPORT void glFramebufferTexture2D (GLenum target, GLenum attachment, GLenum textarget,
  // GLuint texture, GLint level);
  void FramebufferTexture2D(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetFramebufferAttachmentParameteriv (GLenum target, GLenum attachment, GLenum
  // pname, GLint* params);
  Napi::Value GetFramebufferAttachmentParameter(Napi::CallbackInfo const& info);
  // GL_EXPORT GLboolean glIsFramebuffer (GLuint framebuffer);
  Napi::Value IsFramebuffer(Napi::CallbackInfo const& info);
  // GL_EXPORT void glBlitFramebuffer (GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint
  // dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter);
  void BlitFramebuffer(Napi::CallbackInfo const& info);
  // GL_EXPORT void glFramebufferTextureLayer (GLenum target,GLenum attachment, GLuint texture,GLint
  // level,GLint layer);
  void FramebufferTextureLayer(Napi::CallbackInfo const& info);
  // GL_EXPORT void glInvalidateFramebuffer (GLenum target, GLsizei numAttachments, const GLenum*
  // attachments);
  void InvalidateFramebuffer(Napi::CallbackInfo const& info);
  // GL_EXPORT void glInvalidateSubFramebuffer (GLenum target, GLsizei numAttachments, const GLenum*
  // attachments, GLint x, GLint y, GLsizei width, GLsizei height);
  void InvalidateSubFramebuffer(Napi::CallbackInfo const& info);

  ///
  // instanced
  ///
  // GL_EXPORT void glDrawArraysInstanced (GLenum mode, GLint first, GLsizei count, GLsizei
  // primcount);
  void DrawArraysInstanced(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDrawElementsInstanced (GLenum mode, GLsizei count, GLenum type, const void*
  // indices, GLsizei primcount);
  void DrawElementsInstanced(Napi::CallbackInfo const& info);
  // GL_EXPORT void glVertexAttribDivisor (GLuint index, GLuint divisor);
  void VertexAttribDivisor(Napi::CallbackInfo const& info);

  ///
  // program
  ///
  // GL_EXPORT GLuint glCreateProgram (void);
  Napi::Value CreateProgram(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteProgram (GLuint program);
  void DeleteProgram(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetProgramInfoLog (GLuint program, GLsizei bufSize, GLsizei* length, GLchar*
  // infoLog);
  Napi::Value GetProgramInfoLog(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetProgramiv (GLuint program, GLenum pname, GLint* param);
  Napi::Value GetProgramParameter(Napi::CallbackInfo const& info);
  // GL_EXPORT GLboolean glIsProgram (GLuint program);
  Napi::Value IsProgram(Napi::CallbackInfo const& info);
  // GL_EXPORT void glLinkProgram (GLuint program);
  void LinkProgram(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUseProgram (GLuint program);
  void UseProgram(Napi::CallbackInfo const& info);
  // GL_EXPORT void glValidateProgram (GLuint program);
  void ValidateProgram(Napi::CallbackInfo const& info);

  ///
  // query
  ///
  // GL_EXPORT void glBeginQuery (GLenum target, GLuint id);
  void BeginQuery(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGenQueries (GLsizei n, GLuint* ids);
  Napi::Value CreateQuery(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGenQueries (GLsizei n, GLuint* ids);
  Napi::Value CreateQueries(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteQueries (GLsizei n, const GLuint* ids);
  void DeleteQuery(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteQueries (GLsizei n, const GLuint* ids);
  void DeleteQueries(Napi::CallbackInfo const& info);
  // GL_EXPORT void glEndQuery (GLenum target);
  void EndQuery(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetQueryiv (GLenum target, GLenum pname, GLint* params);
  Napi::Value GetQuery(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetQueryObjectuiv (GLuint id, GLenum pname, GLuint* params);
  Napi::Value GetQueryParameter(Napi::CallbackInfo const& info);
  // GL_EXPORT GLboolean glIsQuery (GLuint id);
  Napi::Value IsQuery(Napi::CallbackInfo const& info);
  // GL_EXPORT void glQueryCounter (GLuint id, GLenum target);
  void QueryCounter(Napi::CallbackInfo const& info);

  ///
  // renderbuffer
  ///
  // GL_EXPORT void glCreateRenderbuffers (GLsizei n, GLuint* renderbuffers);
  Napi::Value CreateRenderbuffer(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCreateRenderbuffers (GLsizei n, GLuint* renderbuffers);
  Napi::Value CreateRenderbuffers(Napi::CallbackInfo const& info);
  // GL_EXPORT void glBindRenderbuffer (GLenum target, GLuint renderbuffer);
  void BindRenderbuffer(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteRenderbuffers (GLsizei n, const GLuint* renderbuffers);
  void DeleteRenderbuffer(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteRenderbuffers (GLsizei n, const GLuint* renderbuffers);
  void DeleteRenderbuffers(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetRenderbufferParameteriv (GLenum target, GLenum pname, GLint* params);
  Napi::Value GetRenderbufferParameteriv(Napi::CallbackInfo const& info);
  // GL_EXPORT GLboolean glIsRenderbuffer (GLuint renderbuffer);
  Napi::Value IsRenderbuffer(Napi::CallbackInfo const& info);
  // GL_EXPORT void glRenderbufferStorage (GLenum target, GLenum internalformat, GLsizei width,
  // GLsizei height);
  void RenderbufferStorage(Napi::CallbackInfo const& info);
  // GL_EXPORT void glRenderbufferStorageMultisample (GLenum target, GLsizei samples, GLenum
  // internalformat, GLsizei width, GLsizei height);
  void RenderbufferStorageMultisample(Napi::CallbackInfo const& info);

  ///
  // sampler
  ///
  // GL_EXPORT void glBindSampler (GLuint unit, GLuint sampler);
  void BindSampler(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCreateSamplers (GLsizei n, GLuint* samplers);
  Napi::Value CreateSampler(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCreateSamplers (GLsizei n, GLuint* samplers);
  Napi::Value CreateSamplers(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteSamplers (GLsizei count, const GLuint * samplers);
  void DeleteSampler(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteSamplers (GLsizei count, const GLuint * samplers);
  void DeleteSamplers(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetSamplerParameterfv (GLuint sampler, GLenum pname, GLfloat* params);
  // GL_EXPORT void glGetSamplerParameteriv (GLuint sampler, GLenum pname, GLint* params);
  Napi::Value GetSamplerParameter(Napi::CallbackInfo const& info);
  // GL_EXPORT GLboolean glIsSampler (GLuint sampler);
  Napi::Value IsSampler(Napi::CallbackInfo const& info);
  // GL_EXPORT void glSamplerParameterf (GLuint sampler, GLenum pname, GLfloat param);
  void SamplerParameterf(Napi::CallbackInfo const& info);
  // GL_EXPORT void glSamplerParameteri (GLuint sampler, GLenum pname, GLint param);
  void SamplerParameteri(Napi::CallbackInfo const& info);

  ///
  // shader
  ///
  // GL_EXPORT void glAttachShader (GLuint program, GLuint shader);
  void AttachShader(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCompileShader (GLuint shader);
  void CompileShader(Napi::CallbackInfo const& info);
  // GL_EXPORT GLuint glCreateShader (GLenum type);
  Napi::Value CreateShader(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteShader (GLuint shader);
  void DeleteShader(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDetachShader (GLuint program, GLuint shader);
  void DetachShader(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetAttachedShaders (GLuint program, GLsizei maxCount, GLsizei* count, GLuint*
  // shaders);
  Napi::Value GetAttachedShaders(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetShaderInfoLog (GLuint shader, GLsizei bufSize, GLsizei* length, GLchar*
  // infoLog);
  Napi::Value GetShaderInfoLog(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetShaderiv (GLuint shader, GLenum pname, GLint* param);
  Napi::Value GetShaderParameter(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetShaderPrecisionFormat (GLenum shadertype, GLenum precisiontype, GLint*
  // range, GLint *precision);
  Napi::Value GetShaderPrecisionFormat(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetShaderSource (GLuint obj, GLsizei maxLength, GLsizei* length, GLchar*
  // source);
  Napi::Value GetShaderSource(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetTranslatedShaderSourceANGLE(GLuint shader, GLsizei bufsize, GLsizei*
  // length, GLchar* source);
  Napi::Value GetTranslatedShaderSource(Napi::CallbackInfo const& info);
  // GL_EXPORT GLboolean glIsShader (GLuint shader);
  Napi::Value IsShader(Napi::CallbackInfo const& info);
  // GL_EXPORT void glShaderSource (GLuint shader, GLsizei count, const GLchar *const* string, const
  // GLint* length);
  void ShaderSource(Napi::CallbackInfo const& info);

  ///
  // stencil
  ///
  // GL_EXPORT void glClearStencil (GLint s);
  void ClearStencil(Napi::CallbackInfo const& info);
  // GL_EXPORT void glStencilFunc (GLenum func, GLint ref, GLuint mask);
  void StencilFunc(Napi::CallbackInfo const& info);
  // GL_EXPORT void glStencilFuncSeparate (GLenum frontfunc, GLenum backfunc, GLint ref, GLuint
  // mask);
  void StencilFuncSeparate(Napi::CallbackInfo const& info);
  // GL_EXPORT void glStencilMask (GLuint mask);
  void StencilMask(Napi::CallbackInfo const& info);
  // GL_EXPORT void glStencilMaskSeparate (GLenum face, GLuint mask);
  void StencilMaskSeparate(Napi::CallbackInfo const& info);
  // GL_EXPORT void glStencilOp (GLenum fail, GLenum zfail, GLenum zpass);
  void StencilOp(Napi::CallbackInfo const& info);
  // GL_EXPORT void glStencilOpSeparate (GLenum face, GLenum sfail, GLenum dpfail, GLenum dppass);
  void StencilOpSeparate(Napi::CallbackInfo const& info);

  ///
  // sync
  ///
  // GL_EXPORT GLenum glClientWaitSync (GLsync GLsync, GLbitfield flags,GLuint64 timeout);
  Napi::Value ClientWaitSync(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteSync (GLsync GLsync);
  void DeleteSync(Napi::CallbackInfo const& info);
  // GL_EXPORT GLsync glFenceSync (GLenum condition, GLbitfield flags);
  Napi::Value FenceSync(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetSynciv (GLsync GLsync,GLenum pname,GLsizei bufSize,GLsizei* length, GLint
  // *values);
  Napi::Value GetSyncParameter(Napi::CallbackInfo const& info);
  // GL_EXPORT GLboolean glIsSync (GLsync GLsync);
  Napi::Value IsSync(Napi::CallbackInfo const& info);
  // GL_EXPORT void glWaitSync (GLsync GLsync, GLbitfield flags, GLuint64 timeout);
  void WaitSync(Napi::CallbackInfo const& info);

  ///
  // texture
  ///
  // GL_EXPORT void glActiveTexture (GLenum texture);
  void ActiveTexture(Napi::CallbackInfo const& info);
  // GL_EXPORT void glBindTexture (GLenum target, GLuint texture);
  void BindTexture(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCompressedTexImage2D (GLenum target, GLint level, GLenum internalformat,
  // GLsizei width, GLsizei height, GLint border, GLsizei imageSize, const void *data);
  void CompressedTexImage2D(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCompressedTexImage3D (GLenum target, GLint level, GLenum internalformat,
  // GLsizei width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, const void
  // *data);
  void CompressedTexImage3D(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCompressedTexSubImage2D (GLenum target, GLint level, GLint xoffset, GLint
  // yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const void *data);
  void CompressedTexSubImage2D(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCopyTexImage2D (GLenum target, GLint level, GLenum internalFormat, GLint x,
  // GLint y, GLsizei width, GLsizei height, GLint border);
  void CopyTexImage2D(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCopyTexSubImage2D (GLenum target, GLint level, GLint xoffset, GLint yoffset,
  // GLint x, GLint y, GLsizei width, GLsizei height);
  void CopyTexSubImage2D(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGenTextures (GLsizei n, GLuint* textures);
  Napi::Value CreateTexture(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGenTextures (GLsizei n, GLuint* textures);
  Napi::Value GenTextures(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteTextures (GLsizei n, const GLuint *textures);
  void DeleteTexture(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteTextures (GLsizei n, const GLuint *textures);
  void DeleteTextures(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGenerateMipmap (GLenum target);
  void GenerateMipmap(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetTexParameterfv (GLenum target, GLenum pname, GLfloat *params);
  // GL_EXPORT void glGetTexParameteriv (GLenum target, GLenum pname, GLint *params);
  // GL_EXPORT void glGetTexParameterIiv (GLenum target, GLenum pname, GLint* params);
  // GL_EXPORT void glGetTexParameterIuiv (GLenum target, GLenum pname, GLuint* params);
  Napi::Value GetTexParameter(Napi::CallbackInfo const& info);
  // GL_EXPORT GLboolean glIsTexture (GLuint texture);
  Napi::Value IsTexture(Napi::CallbackInfo const& info);
  // GL_EXPORT void glTexImage2D (GLenum target, GLint level, GLint internalformat, GLsizei width,
  // GLsizei height, GLint border, GLenum format, GLenum type, const void *pixels);
  void TexImage2D(Napi::CallbackInfo const& info);
  // GL_EXPORT void glTexParameterf (GLenum target, GLenum pname, GLfloat param);
  void TexParameterf(Napi::CallbackInfo const& info);
  // GL_EXPORT void glTexParameteri (GLenum target, GLenum pname, GLint param);
  void TexParameteri(Napi::CallbackInfo const& info);
  // GL_EXPORT void glTexSubImage2D (GLenum target, GLint level, GLint xoffset, GLint yoffset,
  // GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels);
  void TexSubImage2D(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCompressedTexSubImage3D (GLenum target, GLint level, GLint xoffset, GLint
  // yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei
  // imageSize, const void *data);
  void CompressedTexSubImage3D(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCopyTexSubImage3D (GLenum target, GLint level, GLint xoffset, GLint yoffset,
  // GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height);
  void CopyTexSubImage3D(Napi::CallbackInfo const& info);
  // GL_EXPORT void glTexImage3D (GLenum target, GLint level, GLint internalFormat, GLsizei width,
  // GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const void *pixels);
  void TexImage3D(Napi::CallbackInfo const& info);
  // GL_EXPORT void glTexStorage2D (GLenum target, GLsizei levels, GLenum internalformat, GLsizei
  // width, GLsizei height);
  void TexStorage2D(Napi::CallbackInfo const& info);
  // GL_EXPORT void glTexStorage3D (GLenum target, GLsizei levels, GLenum internalformat, GLsizei
  // width, GLsizei height, GLsizei depth);
  void TexStorage3D(Napi::CallbackInfo const& info);
  // GL_EXPORT void glTexSubImage3D (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint
  // zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void
  // *pixels);
  void TexSubImage3D(Napi::CallbackInfo const& info);

  ///
  // transform feedback
  ///
  // GL_EXPORT void glBeginTransformFeedback (GLenum primitiveMode);
  void BeginTransformFeedback(Napi::CallbackInfo const& info);
  // GL_EXPORT void glBindTransformFeedback (GLenum target, GLuint id);
  void BindTransformFeedback(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCreateTransformFeedbacks (GLsizei n, GLuint* ids);
  Napi::Value CreateTransformFeedback(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCreateTransformFeedbacks (GLsizei n, GLuint* ids);
  Napi::Value CreateTransformFeedbacks(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteTransformFeedbacks (GLsizei n, const GLuint* ids);
  void DeleteTransformFeedback(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteTransformFeedbacks (GLsizei n, const GLuint* ids);
  void DeleteTransformFeedbacks(Napi::CallbackInfo const& info);
  // GL_EXPORT void glEndTransformFeedback (void);
  void EndTransformFeedback(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetTransformFeedbackVarying (GLuint program, GLuint index, GLsizei bufSize,
  // GLsizei * length, GLsizei * size, GLenum * type, GLchar * name);
  Napi::Value GetTransformFeedbackVarying(Napi::CallbackInfo const& info);
  // GL_EXPORT GLboolean glIsTransformFeedback (GLuint id);
  Napi::Value IsTransformFeedback(Napi::CallbackInfo const& info);
  // GL_EXPORT void glPauseTransformFeedback (void);
  void PauseTransformFeedback(Napi::CallbackInfo const& info);
  // GL_EXPORT void glResumeTransformFeedback (void);
  void ResumeTransformFeedback(Napi::CallbackInfo const& info);
  // GL_EXPORT void glTransformFeedbackVaryings (GLuint program, GLsizei count, const GLchar *const*
  // varyings, GLenum bufferMode);
  void TransformFeedbackVaryings(Napi::CallbackInfo const& info);

  ///
  // uniforms
  ///
  // GL_EXPORT void glGetActiveUniform (GLuint program, GLuint index, GLsizei maxLength, GLsizei*
  // length, GLint* size, GLenum* type, GLchar* name);
  Napi::Value GetActiveUniform(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetUniformfv (GLuint program, GLint location, GLfloat* params);
  // GL_EXPORT void glGetUniformiv (GLuint program, GLint location, GLint* params);
  // GL_EXPORT void glGetUniformuiv (GLuint program, GLint location, GLuint* params);
  // GL_EXPORT void glGetUniformdv (GLuint program, GLint location, GLdouble* params);
  Napi::Value GetUniform(Napi::CallbackInfo const& info);
  // GL_EXPORT GLint glGetUniformLocation (GLuint program, const GLchar* name);
  Napi::Value GetUniformLocation(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform1f (GLint location, GLfloat v0);
  void Uniform1f(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform1fv (GLint location, GLsizei count, const GLfloat* value);
  void Uniform1fv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform1i (GLint location, GLint v0);
  void Uniform1i(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform1iv (GLint location, GLsizei count, const GLint* value);
  void Uniform1iv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform2f (GLint location, GLfloat v0, GLfloat v1);
  void Uniform2f(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform2fv (GLint location, GLsizei count, const GLfloat* value);
  void Uniform2fv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform2i (GLint location, GLint v0, GLint v1);
  void Uniform2i(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform2iv (GLint location, GLsizei count, const GLint* value);
  void Uniform2iv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform3f (GLint location, GLfloat v0, GLfloat v1, GLfloat v2);
  void Uniform3f(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform3fv (GLint location, GLsizei count, const GLfloat* value);
  void Uniform3fv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform3i (GLint location, GLint v0, GLint v1, GLint v2);
  void Uniform3i(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform3iv (GLint location, GLsizei count, const GLint* value);
  void Uniform3iv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform4f (GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3);
  void Uniform4f(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform4fv (GLint location, GLsizei count, const GLfloat* value);
  void Uniform4fv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform4i (GLint location, GLint v0, GLint v1, GLint v2, GLint v3);
  void Uniform4i(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform4iv (GLint location, GLsizei count, const GLint* value);
  void Uniform4iv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniformMatrix2fv (GLint location, GLsizei count, GLboolean transpose, const
  // GLfloat* value);
  void UniformMatrix2fv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniformMatrix3fv (GLint location, GLsizei count, GLboolean transpose, const
  // GLfloat* value);
  void UniformMatrix3fv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniformMatrix4fv (GLint location, GLsizei count, GLboolean transpose, const
  // GLfloat* value);
  void UniformMatrix4fv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniformMatrix2x3fv (GLint location, GLsizei count, GLboolean transpose, const
  // GLfloat *value);
  void UniformMatrix2x3fv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniformMatrix2x4fv (GLint location, GLsizei count, GLboolean transpose, const
  // GLfloat *value);
  void UniformMatrix2x4fv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniformMatrix3x2fv (GLint location, GLsizei count, GLboolean transpose, const
  // GLfloat *value);
  void UniformMatrix3x2fv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniformMatrix3x4fv (GLint location, GLsizei count, GLboolean transpose, const
  // GLfloat *value);
  void UniformMatrix3x4fv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniformMatrix4x2fv (GLint location, GLsizei count, GLboolean transpose, const
  // GLfloat *value);
  void UniformMatrix4x2fv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniformMatrix4x3fv (GLint location, GLsizei count, GLboolean transpose, const
  // GLfloat *value);
  void UniformMatrix4x3fv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform1ui (GLint location, GLuint v0);
  void Uniform1ui(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform1uiv (GLint location, GLsizei count, const GLuint* value);
  void Uniform1uiv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform2ui (GLint location, GLuint v0, GLuint v1);
  void Uniform2ui(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform2uiv (GLint location, GLsizei count, const GLuint* value);
  void Uniform2uiv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform3ui (GLint location, GLuint v0, GLuint v1, GLuint v2);
  void Uniform3ui(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform3uiv (GLint location, GLsizei count, const GLuint* value);
  void Uniform3uiv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform4ui (GLint location, GLuint v0, GLuint v1, GLuint v2, GLuint v3);
  void Uniform4ui(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform4uiv (GLint location, GLsizei count, const GLuint* value);
  void Uniform4uiv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetActiveUniformBlockName (GLuint program, GLuint uniformBlockIndex, GLsizei
  // bufSize, GLsizei* length, GLchar* uniformBlockName);
  Napi::Value GetActiveUniformBlockName(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetActiveUniformBlockiv (GLuint program, GLuint uniformBlockIndex, GLenum
  // pname, GLint* params);
  Napi::Value GetActiveUniformBlockiv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetActiveUniformsiv (GLuint program, GLsizei uniformCount, const GLuint*
  // uniformIndices, GLenum pname, GLint* params);
  Napi::Value GetActiveUniformsiv(Napi::CallbackInfo const& info);
  // GL_EXPORT GLuint glGetUniformBlockIndex (GLuint program, const GLchar* uniformBlockName);
  Napi::Value GetUniformBlockIndex(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetUniformIndices (GLuint program, GLsizei uniformCount, const GLchar* const *
  // uniformNames, GLuint* uniformIndices);
  Napi::Value GetUniformIndices(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniformBlockBinding (GLuint program, GLuint uniformBlockIndex, GLuint
  // uniformBlockBinding);
  void UniformBlockBinding(Napi::CallbackInfo const& info);

  ///
  // vertex array objects
  ///
  // GL_EXPORT void glCreateVertexArrays (GLsizei n, GLuint* arrays);
  Napi::Value CreateVertexArray(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCreateVertexArrays (GLsizei n, GLuint* arrays);
  Napi::Value CreateVertexArrays(Napi::CallbackInfo const& info);
  // GL_EXPORT void glBindVertexArray (GLuint array);
  void BindVertexArray(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteVertexArrays (GLsizei n, const GLuint* arrays);
  void DeleteVertexArray(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteVertexArrays (GLsizei n, const GLuint* arrays);
  void DeleteVertexArrays(Napi::CallbackInfo const& info);
  // GL_EXPORT GLboolean glIsVertexArray (GLuint array);
  Napi::Value IsVertexArray(Napi::CallbackInfo const& info);

  Napi::Value GetContextAttributes(Napi::CallbackInfo const& info);

  Napi::Value GetClearMask_(Napi::CallbackInfo const& info);
  void SetClearMask_(Napi::CallbackInfo const& info, Napi::Value const& value);

  GLbitfield clear_mask_{};
  Napi::ObjectReference context_attributes_;
  // Pixel storage flags
  bool unpack_flip_y_{};
  bool unpack_premultiply_alpha_{};
  GLint unpack_colorspace_conversion_{};
  GLint unpack_alignment_{};
};

}  // namespace nv
