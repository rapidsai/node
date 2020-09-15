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

#include <napi.h>
#include "gl.hpp"

namespace node_webgl {

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

}  // namespace node_webgl

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
