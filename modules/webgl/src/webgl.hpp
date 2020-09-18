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

#pragma once

#include "gl.hpp"

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

class WebGLActiveInfo : public Napi::ObjectWrap<WebGLActiveInfo> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  static Napi::Object New(GLint size, GLuint type, std::string name);

  WebGLActiveInfo(Napi::CallbackInfo const& info);

 private:
  static Napi::FunctionReference constructor;
  Napi::Value ToString(Napi::CallbackInfo const& info);
  Napi::Value GetSize(Napi::CallbackInfo const& info);
  Napi::Value GetType(Napi::CallbackInfo const& info);
  Napi::Value GetName(Napi::CallbackInfo const& info);

  GLint size_{0};
  GLuint type_{0};
  std::string name_{""};
};

class WebGLShaderPrecisionFormat : public Napi::ObjectWrap<WebGLShaderPrecisionFormat> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  static Napi::Object New(GLint rangeMax, GLint rangeMin, GLint precision);

  WebGLShaderPrecisionFormat(Napi::CallbackInfo const& info);

 private:
  static Napi::FunctionReference constructor;
  Napi::Value ToString(Napi::CallbackInfo const& info);
  Napi::Value GetRangeMax(Napi::CallbackInfo const& info);
  Napi::Value GetRangeMin(Napi::CallbackInfo const& info);
  Napi::Value GetPrecision(Napi::CallbackInfo const& info);

  GLint rangeMax_{0};
  GLint rangeMin_{0};
  GLint precision_{0};
};

class WebGLBuffer : public Napi::ObjectWrap<WebGLBuffer> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  static Napi::Object New(GLuint value);

  WebGLBuffer(Napi::CallbackInfo const& info);
  operator GLuint() { return this->value_; }

 private:
  static Napi::FunctionReference constructor;
  Napi::Value ToString(Napi::CallbackInfo const& info);
  Napi::Value GetValue(Napi::CallbackInfo const& info);
  GLuint value_{0};
};

class WebGLContextEvent : public Napi::ObjectWrap<WebGLContextEvent> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  static Napi::Object New(GLuint value);

  WebGLContextEvent(Napi::CallbackInfo const& info);
  operator GLuint() { return this->value_; }

 private:
  static Napi::FunctionReference constructor;
  Napi::Value ToString(Napi::CallbackInfo const& info);
  Napi::Value GetValue(Napi::CallbackInfo const& info);
  GLuint value_{0};
};

class WebGLFramebuffer : public Napi::ObjectWrap<WebGLFramebuffer> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  static Napi::Object New(GLuint value);

  WebGLFramebuffer(Napi::CallbackInfo const& info);
  operator GLuint() { return this->value_; }

 private:
  static Napi::FunctionReference constructor;
  Napi::Value ToString(Napi::CallbackInfo const& info);
  Napi::Value GetValue(Napi::CallbackInfo const& info);
  GLuint value_{0};
};

class WebGLProgram : public Napi::ObjectWrap<WebGLProgram> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  static Napi::Object New(GLuint value);

  WebGLProgram(Napi::CallbackInfo const& info);
  operator GLuint() { return this->value_; }

 private:
  static Napi::FunctionReference constructor;
  Napi::Value ToString(Napi::CallbackInfo const& info);
  Napi::Value GetValue(Napi::CallbackInfo const& info);
  GLuint value_{0};
};

class WebGLQuery : public Napi::ObjectWrap<WebGLQuery> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  static Napi::Object New(GLuint value);

  WebGLQuery(Napi::CallbackInfo const& info);
  operator GLuint() { return this->value_; }

 private:
  static Napi::FunctionReference constructor;
  Napi::Value ToString(Napi::CallbackInfo const& info);
  Napi::Value GetValue(Napi::CallbackInfo const& info);
  GLuint value_{0};
};

class WebGLRenderbuffer : public Napi::ObjectWrap<WebGLRenderbuffer> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  static Napi::Object New(GLuint value);

  WebGLRenderbuffer(Napi::CallbackInfo const& info);
  operator GLuint() { return this->value_; }

 private:
  static Napi::FunctionReference constructor;
  Napi::Value ToString(Napi::CallbackInfo const& info);
  Napi::Value GetValue(Napi::CallbackInfo const& info);
  GLuint value_{0};
};

class WebGLSampler : public Napi::ObjectWrap<WebGLSampler> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  static Napi::Object New(GLuint value);

  WebGLSampler(Napi::CallbackInfo const& info);
  operator GLuint() { return this->value_; }

 private:
  static Napi::FunctionReference constructor;
  Napi::Value ToString(Napi::CallbackInfo const& info);
  Napi::Value GetValue(Napi::CallbackInfo const& info);
  GLuint value_{0};
};

class WebGLShader : public Napi::ObjectWrap<WebGLShader> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  static Napi::Object New(GLuint value);

  WebGLShader(Napi::CallbackInfo const& info);
  operator GLuint() { return this->value_; }

 private:
  static Napi::FunctionReference constructor;
  Napi::Value ToString(Napi::CallbackInfo const& info);
  Napi::Value GetValue(Napi::CallbackInfo const& info);
  GLuint value_{0};
};

class WebGLSync : public Napi::ObjectWrap<WebGLSync> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  static Napi::Object New(GLuint value);

  WebGLSync(Napi::CallbackInfo const& info);
  operator GLuint() { return this->value_; }

 private:
  static Napi::FunctionReference constructor;
  Napi::Value ToString(Napi::CallbackInfo const& info);
  Napi::Value GetValue(Napi::CallbackInfo const& info);
  GLuint value_{0};
};

class WebGLTexture : public Napi::ObjectWrap<WebGLTexture> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  static Napi::Object New(GLuint value);

  WebGLTexture(Napi::CallbackInfo const& info);
  operator GLuint() { return this->value_; }

 private:
  static Napi::FunctionReference constructor;
  Napi::Value ToString(Napi::CallbackInfo const& info);
  Napi::Value GetValue(Napi::CallbackInfo const& info);
  GLuint value_{0};
};

class WebGLTransformFeedback : public Napi::ObjectWrap<WebGLTransformFeedback> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  static Napi::Object New(GLuint value);

  WebGLTransformFeedback(Napi::CallbackInfo const& info);
  operator GLuint() { return this->value_; }

 private:
  static Napi::FunctionReference constructor;
  Napi::Value ToString(Napi::CallbackInfo const& info);
  Napi::Value GetValue(Napi::CallbackInfo const& info);
  GLuint value_{0};
};

class WebGLUniformLocation : public Napi::ObjectWrap<WebGLUniformLocation> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  static Napi::Object New(GLuint value);

  WebGLUniformLocation(Napi::CallbackInfo const& info);
  operator GLuint() { return this->value_; }

 private:
  static Napi::FunctionReference constructor;
  Napi::Value ToString(Napi::CallbackInfo const& info);
  Napi::Value GetValue(Napi::CallbackInfo const& info);
  GLuint value_{0};
};

class WebGLVertexArrayObject : public Napi::ObjectWrap<WebGLVertexArrayObject> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  static Napi::Object New(GLuint value);

  WebGLVertexArrayObject(Napi::CallbackInfo const& info);
  operator GLuint() { return this->value_; }

 private:
  static Napi::FunctionReference constructor;
  Napi::Value ToString(Napi::CallbackInfo const& info);
  Napi::Value GetValue(Napi::CallbackInfo const& info);
  GLuint value_{0};
};

class WebGL2RenderingContext : public Napi::ObjectWrap<WebGL2RenderingContext> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);

  WebGL2RenderingContext(Napi::CallbackInfo const& info);

 private:
  static Napi::FunctionReference constructor;

  ///
  // misc
  ///
  // GL_EXPORT void glClear (GLbitfield mask);
  Napi::Value Clear(Napi::CallbackInfo const& info);
  // GL_EXPORT void glClearColor (GLclampf red, GLclampf green, GLclampf blue, GLclampf alpha);
  Napi::Value ClearColor(Napi::CallbackInfo const& info);
  // GL_EXPORT void glClearDepth (GLclampd depth);
  Napi::Value ClearDepth(Napi::CallbackInfo const& info);
  // GL_EXPORT void glColorMask (GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha);
  Napi::Value ColorMask(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCullFace (GLenum mode);
  Napi::Value CullFace(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDepthFunc (GLenum func);
  Napi::Value DepthFunc(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDepthMask (GLboolean flag);
  Napi::Value DepthMask(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDepthRange (GLclampd zNear, GLclampd zFar);
  Napi::Value DepthRange(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDisable (GLenum cap);
  Napi::Value Disable(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDrawArrays (GLenum mode, GLint first, GLsizei count);
  Napi::Value DrawArrays(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDrawElements (GLenum mode, GLsizei count, GLenum type, const void *indices);
  Napi::Value DrawElements(Napi::CallbackInfo const& info);
  // GL_EXPORT void glEnable (GLenum cap);
  Napi::Value Enable(Napi::CallbackInfo const& info);
  // GL_EXPORT void glFinish (void);
  Napi::Value Finish(Napi::CallbackInfo const& info);
  // GL_EXPORT void glFlush (void);
  Napi::Value Flush(Napi::CallbackInfo const& info);
  // GL_EXPORT void glFrontFace (GLenum mode);
  Napi::Value FrontFace(Napi::CallbackInfo const& info);
  // GL_EXPORT GLenum glGetError (void);
  Napi::Value GetError(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetParameter (GLint pname);
  Napi::Value GetParameter(Napi::CallbackInfo const& info);
  // GL_EXPORT const GLubyte * glGetString (GL_EXTENSIONS);
  Napi::Value GetSupportedExtensions(Napi::CallbackInfo const& info);
  // GL_EXPORT void glHint (GLenum target, GLenum mode);
  Napi::Value Hint(Napi::CallbackInfo const& info);
  // GL_EXPORT GLboolean glIsEnabled (GLenum cap);
  Napi::Value IsEnabled(Napi::CallbackInfo const& info);
  // GL_EXPORT void glLineWidth (GLfloat width);
  Napi::Value LineWidth(Napi::CallbackInfo const& info);
  // GL_EXPORT void glPixelStorei (GLenum pname, GLint param);
  Napi::Value PixelStorei(Napi::CallbackInfo const& info);
  // GL_EXPORT void glPolygonOffset (GLfloat factor, GLfloat units);
  Napi::Value PolygonOffset(Napi::CallbackInfo const& info);
  // GL_EXPORT void glReadPixels (GLint x, GLint y, GLsizei width, GLsizei height, GLenum format,
  // GLenum type, void *pixels);
  Napi::Value ReadPixels(Napi::CallbackInfo const& info);
  // GL_EXPORT void glScissor (GLint x, GLint y, GLsizei width, GLsizei height);
  Napi::Value Scissor(Napi::CallbackInfo const& info);
  // GL_EXPORT void glViewport (GLint x, GLint y, GLsizei width, GLsizei height);
  Napi::Value Viewport(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDrawRangeElements (GLenum mode, GLuint start, GLuint end, GLsizei count,
  // GLenum type, const void *indices);
  Napi::Value DrawRangeElements(Napi::CallbackInfo const& info);
  // GL_EXPORT void glSampleCoverage (GLclampf value, GLboolean invert);
  Napi::Value SampleCoverage(Napi::CallbackInfo const& info);
  // GL_EXPORT GLint glGetFragDataLocation (GLuint program, const GLchar* name);
  Napi::Value GetFragDataLocation(Napi::CallbackInfo const& info);

  ///
  // attrib
  ///

  // GL_EXPORT void glBindAttribLocation (GLuint program, GLuint index, const GLchar* name);
  Napi::Value BindAttribLocation(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDisableVertexAttribArray (GLuint index);
  Napi::Value DisableVertexAttribArray(Napi::CallbackInfo const& info);
  // GL_EXPORT void glEnableVertexAttribArray (GLuint index);
  Napi::Value EnableVertexAttribArray(Napi::CallbackInfo const& info);
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
  Napi::Value VertexAttrib1f(Napi::CallbackInfo const& info);
  // GL_EXPORT void glVertexAttrib1fv (GLuint index, const GLfloat* v);
  Napi::Value VertexAttrib1fv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glVertexAttrib2f (GLuint index, GLfloat x, GLfloat y);
  Napi::Value VertexAttrib2f(Napi::CallbackInfo const& info);
  // GL_EXPORT void glVertexAttrib2fv (GLuint index, const GLfloat* v);
  Napi::Value VertexAttrib2fv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glVertexAttrib3f (GLuint index, GLfloat x, GLfloat y, GLfloat z);
  Napi::Value VertexAttrib3f(Napi::CallbackInfo const& info);
  // GL_EXPORT void glVertexAttrib3fv (GLuint index, const GLfloat* v);
  Napi::Value VertexAttrib3fv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glVertexAttrib4f (GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
  Napi::Value VertexAttrib4f(Napi::CallbackInfo const& info);
  // GL_EXPORT void glVertexAttrib4fv (GLuint index, const GLfloat* v);
  Napi::Value VertexAttrib4fv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glVertexAttribPointer (GLuint index, GLint size, GLenum type, GLboolean
  // normalized, GLsizei stride, const void* pointer);
  Napi::Value VertexAttribPointer(Napi::CallbackInfo const& info);
  // GL_EXPORT void glVertexAttribI4i (GLuint index, GLint v0, GLint v1, GLint v2, GLint v3);
  Napi::Value VertexAttribI4i(Napi::CallbackInfo const& info);
  // GL_EXPORT void glVertexAttribI4iv (GLuint index, const GLint* v0);
  Napi::Value VertexAttribI4iv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glVertexAttribI4ui (GLuint index, GLuint v0, GLuint v1, GLuint v2, GLuint v3);
  Napi::Value VertexAttribI4ui(Napi::CallbackInfo const& info);
  // GL_EXPORT void glVertexAttribI4uiv (GLuint index, const GLuint* v0);
  Napi::Value VertexAttribI4uiv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glVertexAttribIPointer (GLuint index, GLint size, GLenum type, GLsizei stride,
  // const void*pointer);
  Napi::Value VertexAttribIPointer(Napi::CallbackInfo const& info);

  ///
  // blend
  ///
  // GL_EXPORT void glBlendColor (GLclampf red, GLclampf green, GLclampf blue, GLclampf alpha);
  Napi::Value BlendColor(Napi::CallbackInfo const& info);
  // GL_EXPORT void glBlendEquation (GLenum mode);
  Napi::Value BlendEquation(Napi::CallbackInfo const& info);
  // GL_EXPORT void glBlendEquationSeparate (GLenum modeRGB, GLenum modeAlpha);
  Napi::Value BlendEquationSeparate(Napi::CallbackInfo const& info);
  // GL_EXPORT void glBlendFunc (GLenum sfactor, GLenum dfactor);
  Napi::Value BlendFunc(Napi::CallbackInfo const& info);
  // GL_EXPORT void glBlendFuncSeparate (GLenum sfactorRGB, GLenum dfactorRGB, GLenum sfactorAlpha,
  // GLenum dfactorAlpha);
  Napi::Value BlendFuncSeparate(Napi::CallbackInfo const& info);

  ///
  // buffer
  ///
  // GL_EXPORT void glBindBuffer (GLenum target, GLuint buffer);
  Napi::Value BindBuffer(Napi::CallbackInfo const& info);
  // GL_EXPORT void glBufferData (GLenum target, GLsizeiptr size, const void* data, GLenum usage);
  Napi::Value BufferData(Napi::CallbackInfo const& info);
  // GL_EXPORT void glBufferSubData (GLenum target, GLintptr offset, GLsizeiptr size, const void*
  // data);
  Napi::Value BufferSubData(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCreateBuffers (GLsizei n, GLuint* buffers);
  Napi::Value CreateBuffer(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCreateBuffers (GLsizei n, GLuint* buffers);
  Napi::Value CreateBuffers(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteBuffers (GLsizei n, const GLuint* buffers);
  Napi::Value DeleteBuffer(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteBuffers (GLsizei n, const GLuint* buffers);
  Napi::Value DeleteBuffers(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetBufferParameter (GLenum target, GLenum pname, GLint* params);
  Napi::Value GetBufferParameter(Napi::CallbackInfo const& info);
  // GL_EXPORT GLboolean glIsBuffer (GLuint buffer);
  Napi::Value IsBuffer(Napi::CallbackInfo const& info);
  // GL_EXPORT void glBindBufferBase (GLenum target, GLuint index, GLuint buffer);
  Napi::Value BindBufferBase(Napi::CallbackInfo const& info);
  // GL_EXPORT void glBindBufferRange (GLenum target, GLuint index, GLuint buffer, GLintptr offset,
  // GLsizeiptr size);
  Napi::Value BindBufferRange(Napi::CallbackInfo const& info);
  // GL_EXPORT void glClearBufferfv (GLenum buffer, GLint drawBuffer, const GLfloat* value);
  Napi::Value ClearBufferfv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glClearBufferiv (GLenum buffer, GLint drawBuffer, const GLint* value);
  Napi::Value ClearBufferiv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glClearBufferuiv (GLenum buffer, GLint drawBuffer, const GLuint* value);
  Napi::Value ClearBufferuiv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glClearBufferfi (GLenum buffer, GLint drawBuffer, GLfloat depth, GLint stencil);
  Napi::Value ClearBufferfi(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCopyBufferSubData (GLenum readtarget, GLenum writetarget, GLintptr readoffset,
  // GLintptr writeoffset, GLsizeiptr size);
  Napi::Value CopyBufferSubData(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDrawBuffers (GLsizei n, const GLenum* bufs);
  Napi::Value DrawBuffers(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetBufferSubData (GLenum target, GLintptr offset, GLsizeiptr size, void*
  // data);
  Napi::Value GetBufferSubData(Napi::CallbackInfo const& info);
  // GL_EXPORT void glReadBuffer (GLenum mode);
  Napi::Value ReadBuffer(Napi::CallbackInfo const& info);

  ///
  // framebuffer
  ///
  // GL_EXPORT void glBindFramebuffer (GLenum target, GLuint framebuffer);
  Napi::Value BindFramebuffer(Napi::CallbackInfo const& info);
  // GL_EXPORT GLenum glCheckFramebufferStatus (GLenum target);
  Napi::Value CheckFramebufferStatus(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCreateFramebuffers (GLsizei n, GLuint* framebuffers);
  Napi::Value CreateFramebuffer(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCreateFramebuffers (GLsizei n, GLuint* framebuffers);
  Napi::Value CreateFramebuffers(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteFramebuffers (GLsizei n, const GLuint* framebuffers);
  Napi::Value DeleteFramebuffer(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteFramebuffers (GLsizei n, const GLuint* framebuffers);
  Napi::Value DeleteFramebuffers(Napi::CallbackInfo const& info);
  // GL_EXPORT void glFramebufferRenderbuffer (GLenum target, GLenum attachment, GLenum
  // renderbuffertarget, GLuint renderbuffer);
  Napi::Value FramebufferRenderbuffer(Napi::CallbackInfo const& info);
  // GL_EXPORT void glFramebufferTexture2D (GLenum target, GLenum attachment, GLenum textarget,
  // GLuint texture, GLint level);
  Napi::Value FramebufferTexture2D(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetFramebufferAttachmentParameteriv (GLenum target, GLenum attachment, GLenum
  // pname, GLint* params);
  Napi::Value GetFramebufferAttachmentParameter(Napi::CallbackInfo const& info);
  // GL_EXPORT GLboolean glIsFramebuffer (GLuint framebuffer);
  Napi::Value IsFramebuffer(Napi::CallbackInfo const& info);
  // GL_EXPORT void glBlitFramebuffer (GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint
  // dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter);
  Napi::Value BlitFramebuffer(Napi::CallbackInfo const& info);
  // GL_EXPORT void glFramebufferTextureLayer (GLenum target,GLenum attachment, GLuint texture,GLint
  // level,GLint layer);
  Napi::Value FramebufferTextureLayer(Napi::CallbackInfo const& info);
  // GL_EXPORT void glInvalidateFramebuffer (GLenum target, GLsizei numAttachments, const GLenum*
  // attachments);
  Napi::Value InvalidateFramebuffer(Napi::CallbackInfo const& info);
  // GL_EXPORT void glInvalidateSubFramebuffer (GLenum target, GLsizei numAttachments, const GLenum*
  // attachments, GLint x, GLint y, GLsizei width, GLsizei height);
  Napi::Value InvalidateSubFramebuffer(Napi::CallbackInfo const& info);

  ///
  // instanced
  ///
  // GL_EXPORT void glDrawArraysInstanced (GLenum mode, GLint first, GLsizei count, GLsizei
  // primcount);
  Napi::Value DrawArraysInstanced(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDrawElementsInstanced (GLenum mode, GLsizei count, GLenum type, const void*
  // indices, GLsizei primcount);
  Napi::Value DrawElementsInstanced(Napi::CallbackInfo const& info);
  // GL_EXPORT void glVertexAttribDivisor (GLuint index, GLuint divisor);
  Napi::Value VertexAttribDivisor(Napi::CallbackInfo const& info);

  ///
  // program
  ///
  // GL_EXPORT GLuint glCreateProgram (void);
  Napi::Value CreateProgram(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteProgram (GLuint program);
  Napi::Value DeleteProgram(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetProgramInfoLog (GLuint program, GLsizei bufSize, GLsizei* length, GLchar*
  // infoLog);
  Napi::Value GetProgramInfoLog(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetProgramiv (GLuint program, GLenum pname, GLint* param);
  Napi::Value GetProgramParameter(Napi::CallbackInfo const& info);
  // GL_EXPORT GLboolean glIsProgram (GLuint program);
  Napi::Value IsProgram(Napi::CallbackInfo const& info);
  // GL_EXPORT void glLinkProgram (GLuint program);
  Napi::Value LinkProgram(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUseProgram (GLuint program);
  Napi::Value UseProgram(Napi::CallbackInfo const& info);
  // GL_EXPORT void glValidateProgram (GLuint program);
  Napi::Value ValidateProgram(Napi::CallbackInfo const& info);

  ///
  // query
  ///
  // GL_EXPORT void glBeginQuery (GLenum target, GLuint id);
  Napi::Value BeginQuery(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGenQueries (GLsizei n, GLuint* ids);
  Napi::Value CreateQuery(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGenQueries (GLsizei n, GLuint* ids);
  Napi::Value CreateQueries(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteQueries (GLsizei n, const GLuint* ids);
  Napi::Value DeleteQuery(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteQueries (GLsizei n, const GLuint* ids);
  Napi::Value DeleteQueries(Napi::CallbackInfo const& info);
  // GL_EXPORT void glEndQuery (GLenum target);
  Napi::Value EndQuery(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetQueryiv (GLenum target, GLenum pname, GLint* params);
  Napi::Value GetQuery(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetQueryObjectuiv (GLuint id, GLenum pname, GLuint* params);
  Napi::Value GetQueryParameter(Napi::CallbackInfo const& info);
  // GL_EXPORT GLboolean glIsQuery (GLuint id);
  Napi::Value IsQuery(Napi::CallbackInfo const& info);

  ///
  // renderbuffer
  ///
  // GL_EXPORT void glCreateRenderbuffers (GLsizei n, GLuint* renderbuffers);
  Napi::Value CreateRenderbuffer(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCreateRenderbuffers (GLsizei n, GLuint* renderbuffers);
  Napi::Value CreateRenderbuffers(Napi::CallbackInfo const& info);
  // GL_EXPORT void glBindRenderbuffer (GLenum target, GLuint renderbuffer);
  Napi::Value BindRenderbuffer(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteRenderbuffers (GLsizei n, const GLuint* renderbuffers);
  Napi::Value DeleteRenderbuffer(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteRenderbuffers (GLsizei n, const GLuint* renderbuffers);
  Napi::Value DeleteRenderbuffers(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetRenderbufferParameteriv (GLenum target, GLenum pname, GLint* params);
  Napi::Value GetRenderbufferParameteriv(Napi::CallbackInfo const& info);
  // GL_EXPORT GLboolean glIsRenderbuffer (GLuint renderbuffer);
  Napi::Value IsRenderbuffer(Napi::CallbackInfo const& info);
  // GL_EXPORT void glRenderbufferStorage (GLenum target, GLenum internalformat, GLsizei width,
  // GLsizei height);
  Napi::Value RenderbufferStorage(Napi::CallbackInfo const& info);
  // GL_EXPORT void glRenderbufferStorageMultisample (GLenum target, GLsizei samples, GLenum
  // internalformat, GLsizei width, GLsizei height);
  Napi::Value RenderbufferStorageMultisample(Napi::CallbackInfo const& info);

  ///
  // sampler
  ///
  // GL_EXPORT void glBindSampler (GLuint unit, GLuint sampler);
  Napi::Value BindSampler(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCreateSamplers (GLsizei n, GLuint* samplers);
  Napi::Value CreateSampler(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCreateSamplers (GLsizei n, GLuint* samplers);
  Napi::Value CreateSamplers(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteSamplers (GLsizei count, const GLuint * samplers);
  Napi::Value DeleteSampler(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteSamplers (GLsizei count, const GLuint * samplers);
  Napi::Value DeleteSamplers(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetSamplerParameterfv (GLuint sampler, GLenum pname, GLfloat* params);
  // GL_EXPORT void glGetSamplerParameteriv (GLuint sampler, GLenum pname, GLint* params);
  Napi::Value GetSamplerParameter(Napi::CallbackInfo const& info);
  // GL_EXPORT GLboolean glIsSampler (GLuint sampler);
  Napi::Value IsSampler(Napi::CallbackInfo const& info);
  // GL_EXPORT void glSamplerParameterf (GLuint sampler, GLenum pname, GLfloat param);
  Napi::Value SamplerParameterf(Napi::CallbackInfo const& info);
  // GL_EXPORT void glSamplerParameteri (GLuint sampler, GLenum pname, GLint param);
  Napi::Value SamplerParameteri(Napi::CallbackInfo const& info);

  ///
  // shader
  ///
  // GL_EXPORT void glAttachShader (GLuint program, GLuint shader);
  Napi::Value AttachShader(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCompileShader (GLuint shader);
  Napi::Value CompileShader(Napi::CallbackInfo const& info);
  // GL_EXPORT GLuint glCreateShader (GLenum type);
  Napi::Value CreateShader(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteShader (GLuint shader);
  Napi::Value DeleteShader(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDetachShader (GLuint program, GLuint shader);
  Napi::Value DetachShader(Napi::CallbackInfo const& info);
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
  Napi::Value ShaderSource(Napi::CallbackInfo const& info);

  ///
  // stencil
  ///
  // GL_EXPORT void glClearStencil (GLint s);
  Napi::Value ClearStencil(Napi::CallbackInfo const& info);
  // GL_EXPORT void glStencilFunc (GLenum func, GLint ref, GLuint mask);
  Napi::Value StencilFunc(Napi::CallbackInfo const& info);
  // GL_EXPORT void glStencilFuncSeparate (GLenum frontfunc, GLenum backfunc, GLint ref, GLuint
  // mask);
  Napi::Value StencilFuncSeparate(Napi::CallbackInfo const& info);
  // GL_EXPORT void glStencilMask (GLuint mask);
  Napi::Value StencilMask(Napi::CallbackInfo const& info);
  // GL_EXPORT void glStencilMaskSeparate (GLenum face, GLuint mask);
  Napi::Value StencilMaskSeparate(Napi::CallbackInfo const& info);
  // GL_EXPORT void glStencilOp (GLenum fail, GLenum zfail, GLenum zpass);
  Napi::Value StencilOp(Napi::CallbackInfo const& info);
  // GL_EXPORT void glStencilOpSeparate (GLenum face, GLenum sfail, GLenum dpfail, GLenum dppass);
  Napi::Value StencilOpSeparate(Napi::CallbackInfo const& info);

  ///
  // sync
  ///
  // GL_EXPORT GLenum glClientWaitSync (GLsync GLsync, GLbitfield flags,GLuint64 timeout);
  Napi::Value ClientWaitSync(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteSync (GLsync GLsync);
  Napi::Value DeleteSync(Napi::CallbackInfo const& info);
  // GL_EXPORT GLsync glFenceSync (GLenum condition, GLbitfield flags);
  Napi::Value FenceSync(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetSynciv (GLsync GLsync,GLenum pname,GLsizei bufSize,GLsizei* length, GLint
  // *values);
  Napi::Value GetSyncParameter(Napi::CallbackInfo const& info);
  // GL_EXPORT GLboolean glIsSync (GLsync GLsync);
  Napi::Value IsSync(Napi::CallbackInfo const& info);
  // GL_EXPORT void glWaitSync (GLsync GLsync, GLbitfield flags, GLuint64 timeout);
  Napi::Value WaitSync(Napi::CallbackInfo const& info);

  ///
  // texture
  ///
  // GL_EXPORT void glActiveTexture (GLenum texture);
  Napi::Value ActiveTexture(Napi::CallbackInfo const& info);
  // GL_EXPORT void glBindTexture (GLenum target, GLuint texture);
  Napi::Value BindTexture(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCompressedTexImage2D (GLenum target, GLint level, GLenum internalformat,
  // GLsizei width, GLsizei height, GLint border, GLsizei imageSize, const void *data);
  Napi::Value CompressedTexImage2D(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCompressedTexImage3D (GLenum target, GLint level, GLenum internalformat,
  // GLsizei width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, const void
  // *data);
  Napi::Value CompressedTexImage3D(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCompressedTexSubImage2D (GLenum target, GLint level, GLint xoffset, GLint
  // yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const void *data);
  Napi::Value CompressedTexSubImage2D(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCopyTexImage2D (GLenum target, GLint level, GLenum internalFormat, GLint x,
  // GLint y, GLsizei width, GLsizei height, GLint border);
  Napi::Value CopyTexImage2D(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCopyTexSubImage2D (GLenum target, GLint level, GLint xoffset, GLint yoffset,
  // GLint x, GLint y, GLsizei width, GLsizei height);
  Napi::Value CopyTexSubImage2D(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGenTextures (GLsizei n, GLuint* textures);
  Napi::Value CreateTexture(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGenTextures (GLsizei n, GLuint* textures);
  Napi::Value GenTextures(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteTextures (GLsizei n, const GLuint *textures);
  Napi::Value DeleteTexture(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteTextures (GLsizei n, const GLuint *textures);
  Napi::Value DeleteTextures(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGenerateMipmap (GLenum target);
  Napi::Value GenerateMipmap(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetTexParameterfv (GLenum target, GLenum pname, GLfloat *params);
  // GL_EXPORT void glGetTexParameteriv (GLenum target, GLenum pname, GLint *params);
  // GL_EXPORT void glGetTexParameterIiv (GLenum target, GLenum pname, GLint* params);
  // GL_EXPORT void glGetTexParameterIuiv (GLenum target, GLenum pname, GLuint* params);
  Napi::Value GetTexParameter(Napi::CallbackInfo const& info);
  // GL_EXPORT GLboolean glIsTexture (GLuint texture);
  Napi::Value IsTexture(Napi::CallbackInfo const& info);
  // GL_EXPORT void glTexImage2D (GLenum target, GLint level, GLint internalformat, GLsizei width,
  // GLsizei height, GLint border, GLenum format, GLenum type, const void *pixels);
  Napi::Value TexImage2D(Napi::CallbackInfo const& info);
  // GL_EXPORT void glTexParameterf (GLenum target, GLenum pname, GLfloat param);
  Napi::Value TexParameterf(Napi::CallbackInfo const& info);
  // GL_EXPORT void glTexParameteri (GLenum target, GLenum pname, GLint param);
  Napi::Value TexParameteri(Napi::CallbackInfo const& info);
  // GL_EXPORT void glTexSubImage2D (GLenum target, GLint level, GLint xoffset, GLint yoffset,
  // GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels);
  Napi::Value TexSubImage2D(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCompressedTexSubImage3D (GLenum target, GLint level, GLint xoffset, GLint
  // yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei
  // imageSize, const void *data);
  Napi::Value CompressedTexSubImage3D(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCopyTexSubImage3D (GLenum target, GLint level, GLint xoffset, GLint yoffset,
  // GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height);
  Napi::Value CopyTexSubImage3D(Napi::CallbackInfo const& info);
  // GL_EXPORT void glTexImage3D (GLenum target, GLint level, GLint internalFormat, GLsizei width,
  // GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const void *pixels);
  Napi::Value TexImage3D(Napi::CallbackInfo const& info);
  // GL_EXPORT void glTexStorage2D (GLenum target, GLsizei levels, GLenum internalformat, GLsizei
  // width, GLsizei height);
  Napi::Value TexStorage2D(Napi::CallbackInfo const& info);
  // GL_EXPORT void glTexStorage3D (GLenum target, GLsizei levels, GLenum internalformat, GLsizei
  // width, GLsizei height, GLsizei depth);
  Napi::Value TexStorage3D(Napi::CallbackInfo const& info);
  // GL_EXPORT void glTexSubImage3D (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint
  // zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void
  // *pixels);
  Napi::Value TexSubImage3D(Napi::CallbackInfo const& info);

  ///
  // transform feedback
  ///
  // GL_EXPORT void glBeginTransformFeedback (GLenum primitiveMode);
  Napi::Value BeginTransformFeedback(Napi::CallbackInfo const& info);
  // GL_EXPORT void glBindTransformFeedback (GLenum target, GLuint id);
  Napi::Value BindTransformFeedback(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCreateTransformFeedbacks (GLsizei n, GLuint* ids);
  Napi::Value CreateTransformFeedback(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCreateTransformFeedbacks (GLsizei n, GLuint* ids);
  Napi::Value CreateTransformFeedbacks(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteTransformFeedbacks (GLsizei n, const GLuint* ids);
  Napi::Value DeleteTransformFeedback(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteTransformFeedbacks (GLsizei n, const GLuint* ids);
  Napi::Value DeleteTransformFeedbacks(Napi::CallbackInfo const& info);
  // GL_EXPORT void glEndTransformFeedback (void);
  Napi::Value EndTransformFeedback(Napi::CallbackInfo const& info);
  // GL_EXPORT void glGetTransformFeedbackVarying (GLuint program, GLuint index, GLsizei bufSize,
  // GLsizei * length, GLsizei * size, GLenum * type, GLchar * name);
  Napi::Value GetTransformFeedbackVarying(Napi::CallbackInfo const& info);
  // GL_EXPORT GLboolean glIsTransformFeedback (GLuint id);
  Napi::Value IsTransformFeedback(Napi::CallbackInfo const& info);
  // GL_EXPORT void glPauseTransformFeedback (void);
  Napi::Value PauseTransformFeedback(Napi::CallbackInfo const& info);
  // GL_EXPORT void glResumeTransformFeedback (void);
  Napi::Value ResumeTransformFeedback(Napi::CallbackInfo const& info);
  // GL_EXPORT void glTransformFeedbackVaryings (GLuint program, GLsizei count, const GLchar *const*
  // varyings, GLenum bufferMode);
  Napi::Value TransformFeedbackVaryings(Napi::CallbackInfo const& info);

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
  Napi::Value Uniform1f(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform1fv (GLint location, GLsizei count, const GLfloat* value);
  Napi::Value Uniform1fv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform1i (GLint location, GLint v0);
  Napi::Value Uniform1i(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform1iv (GLint location, GLsizei count, const GLint* value);
  Napi::Value Uniform1iv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform2f (GLint location, GLfloat v0, GLfloat v1);
  Napi::Value Uniform2f(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform2fv (GLint location, GLsizei count, const GLfloat* value);
  Napi::Value Uniform2fv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform2i (GLint location, GLint v0, GLint v1);
  Napi::Value Uniform2i(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform2iv (GLint location, GLsizei count, const GLint* value);
  Napi::Value Uniform2iv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform3f (GLint location, GLfloat v0, GLfloat v1, GLfloat v2);
  Napi::Value Uniform3f(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform3fv (GLint location, GLsizei count, const GLfloat* value);
  Napi::Value Uniform3fv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform3i (GLint location, GLint v0, GLint v1, GLint v2);
  Napi::Value Uniform3i(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform3iv (GLint location, GLsizei count, const GLint* value);
  Napi::Value Uniform3iv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform4f (GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3);
  Napi::Value Uniform4f(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform4fv (GLint location, GLsizei count, const GLfloat* value);
  Napi::Value Uniform4fv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform4i (GLint location, GLint v0, GLint v1, GLint v2, GLint v3);
  Napi::Value Uniform4i(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform4iv (GLint location, GLsizei count, const GLint* value);
  Napi::Value Uniform4iv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniformMatrix2fv (GLint location, GLsizei count, GLboolean transpose, const
  // GLfloat* value);
  Napi::Value UniformMatrix2fv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniformMatrix3fv (GLint location, GLsizei count, GLboolean transpose, const
  // GLfloat* value);
  Napi::Value UniformMatrix3fv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniformMatrix4fv (GLint location, GLsizei count, GLboolean transpose, const
  // GLfloat* value);
  Napi::Value UniformMatrix4fv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniformMatrix2x3fv (GLint location, GLsizei count, GLboolean transpose, const
  // GLfloat *value);
  Napi::Value UniformMatrix2x3fv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniformMatrix2x4fv (GLint location, GLsizei count, GLboolean transpose, const
  // GLfloat *value);
  Napi::Value UniformMatrix2x4fv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniformMatrix3x2fv (GLint location, GLsizei count, GLboolean transpose, const
  // GLfloat *value);
  Napi::Value UniformMatrix3x2fv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniformMatrix3x4fv (GLint location, GLsizei count, GLboolean transpose, const
  // GLfloat *value);
  Napi::Value UniformMatrix3x4fv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniformMatrix4x2fv (GLint location, GLsizei count, GLboolean transpose, const
  // GLfloat *value);
  Napi::Value UniformMatrix4x2fv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniformMatrix4x3fv (GLint location, GLsizei count, GLboolean transpose, const
  // GLfloat *value);
  Napi::Value UniformMatrix4x3fv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform1ui (GLint location, GLuint v0);
  Napi::Value Uniform1ui(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform1uiv (GLint location, GLsizei count, const GLuint* value);
  Napi::Value Uniform1uiv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform2ui (GLint location, GLuint v0, GLuint v1);
  Napi::Value Uniform2ui(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform2uiv (GLint location, GLsizei count, const GLuint* value);
  Napi::Value Uniform2uiv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform3ui (GLint location, GLuint v0, GLuint v1, GLuint v2);
  Napi::Value Uniform3ui(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform3uiv (GLint location, GLsizei count, const GLuint* value);
  Napi::Value Uniform3uiv(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform4ui (GLint location, GLuint v0, GLuint v1, GLuint v2, GLuint v3);
  Napi::Value Uniform4ui(Napi::CallbackInfo const& info);
  // GL_EXPORT void glUniform4uiv (GLint location, GLsizei count, const GLuint* value);
  Napi::Value Uniform4uiv(Napi::CallbackInfo const& info);
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
  Napi::Value UniformBlockBinding(Napi::CallbackInfo const& info);

  ///
  // vertex array objects
  ///
  // GL_EXPORT void glCreateVertexArrays (GLsizei n, GLuint* arrays);
  Napi::Value CreateVertexArray(Napi::CallbackInfo const& info);
  // GL_EXPORT void glCreateVertexArrays (GLsizei n, GLuint* arrays);
  Napi::Value CreateVertexArrays(Napi::CallbackInfo const& info);
  // GL_EXPORT void glBindVertexArray (GLuint array);
  Napi::Value BindVertexArray(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteVertexArrays (GLsizei n, const GLuint* arrays);
  Napi::Value DeleteVertexArray(Napi::CallbackInfo const& info);
  // GL_EXPORT void glDeleteVertexArrays (GLsizei n, const GLuint* arrays);
  Napi::Value DeleteVertexArrays(Napi::CallbackInfo const& info);
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
