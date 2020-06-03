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

#include <node_webgl/webgl.hpp>

namespace node_webgl {

Napi::FunctionReference WebGLActiveInfo::constructor;

WebGLActiveInfo::WebGLActiveInfo(Napi::CallbackInfo const& info)
  : Napi::ObjectWrap<WebGLActiveInfo>(info){};

Napi::Object WebGLActiveInfo::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor =
    DefineClass(env,
                "WebGLActiveInfo",
                {
                  InstanceAccessor("size", &WebGLActiveInfo::GetSize, nullptr, napi_enumerable),
                  InstanceAccessor("type", &WebGLActiveInfo::GetType, nullptr, napi_enumerable),
                  InstanceAccessor("name", &WebGLActiveInfo::GetName, nullptr, napi_enumerable),
                  InstanceMethod("toString", &WebGLActiveInfo::ToString),
                });
  WebGLActiveInfo::constructor = Napi::Persistent(ctor);
  WebGLActiveInfo::constructor.SuppressDestruct();
  // exports.Set("WebGLActiveInfo", ctor);
  return exports;
};

Napi::Object WebGLActiveInfo::New(GLint size, GLuint type, std::string name) {
  auto obj                            = WebGLActiveInfo::constructor.New({});
  WebGLActiveInfo::Unwrap(obj)->size_ = size;
  WebGLActiveInfo::Unwrap(obj)->type_ = type;
  WebGLActiveInfo::Unwrap(obj)->name_ = name;
  return obj;
};

Napi::Value WebGLActiveInfo::GetSize(Napi::CallbackInfo const& info) {
  return Napi::Number::New(this->Env(), this->size_);
}

Napi::Value WebGLActiveInfo::GetType(Napi::CallbackInfo const& info) {
  return Napi::Number::New(this->Env(), this->type_);
}

Napi::Value WebGLActiveInfo::GetName(Napi::CallbackInfo const& info) {
  return Napi::String::New(this->Env(), this->name_);
}

Napi::Value WebGLActiveInfo::ToString(Napi::CallbackInfo const& info) {
  return Napi::String::New(this->Env(),
                           std::string{"[ WebGLActiveInfo"} + " size=" + std::to_string(size_) +
                             " type=" + std::to_string(type_) + " name='" + name_ + "' ]");
}

Napi::FunctionReference WebGLShaderPrecisionFormat::constructor;

WebGLShaderPrecisionFormat::WebGLShaderPrecisionFormat(Napi::CallbackInfo const& info)
  : Napi::ObjectWrap<WebGLShaderPrecisionFormat>(info){};

Napi::Object WebGLShaderPrecisionFormat::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor = DefineClass(
    env,
    "WebGLShaderPrecisionFormat",
    {
      InstanceAccessor("size", &WebGLShaderPrecisionFormat::GetRangeMax, nullptr, napi_enumerable),
      InstanceAccessor("type", &WebGLShaderPrecisionFormat::GetRangeMin, nullptr, napi_enumerable),
      InstanceAccessor("name", &WebGLShaderPrecisionFormat::GetPrecision, nullptr, napi_enumerable),
      InstanceMethod("toString", &WebGLShaderPrecisionFormat::ToString),
    });
  WebGLShaderPrecisionFormat::constructor = Napi::Persistent(ctor);
  WebGLShaderPrecisionFormat::constructor.SuppressDestruct();
  // exports.Set("WebGLShaderPrecisionFormat", ctor);
  return exports;
};

Napi::Object WebGLShaderPrecisionFormat::New(GLint rangeMax, GLint rangeMin, GLint precision) {
  auto obj = WebGLShaderPrecisionFormat::constructor.New({});
  WebGLShaderPrecisionFormat::Unwrap(obj)->rangeMax_  = rangeMax;
  WebGLShaderPrecisionFormat::Unwrap(obj)->rangeMin_  = rangeMin;
  WebGLShaderPrecisionFormat::Unwrap(obj)->precision_ = precision;
  return obj;
};

Napi::Value WebGLShaderPrecisionFormat::GetRangeMax(Napi::CallbackInfo const& info) {
  return Napi::Number::New(this->Env(), this->rangeMax_);
}

Napi::Value WebGLShaderPrecisionFormat::GetRangeMin(Napi::CallbackInfo const& info) {
  return Napi::Number::New(this->Env(), this->rangeMin_);
}

Napi::Value WebGLShaderPrecisionFormat::GetPrecision(Napi::CallbackInfo const& info) {
  return Napi::Number::New(this->Env(), this->precision_);
}

Napi::Value WebGLShaderPrecisionFormat::ToString(Napi::CallbackInfo const& info) {
  return Napi::String::New(
    this->Env(),
    std::string{"[ WebGLShaderPrecisionFormat"} + " rangeMax=" + std::to_string(rangeMax_) +
      " rangeMin=" + std::to_string(rangeMin_) + " precision=" + std::to_string(precision_) + " ]");
}

Napi::FunctionReference WebGLBuffer::constructor;

WebGLBuffer::WebGLBuffer(Napi::CallbackInfo const& info) : Napi::ObjectWrap<WebGLBuffer>(info){};

Napi::Object WebGLBuffer::New(GLuint value) {
  auto obj                         = WebGLBuffer::constructor.New({});
  WebGLBuffer::Unwrap(obj)->value_ = value;
  return obj;
};

Napi::Object WebGLBuffer::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor =
    DefineClass(env,
                "WebGLBuffer",
                {
                  InstanceAccessor("_", &WebGLBuffer::GetValue, nullptr, napi_enumerable),
                  InstanceMethod("toString", &WebGLBuffer::ToString),
                });
  WebGLBuffer::constructor = Napi::Persistent(ctor);
  WebGLBuffer::constructor.SuppressDestruct();
  // exports.Set("WebGLBuffer", ctor);
  return exports;
};

Napi::Value WebGLBuffer::ToString(Napi::CallbackInfo const& info) {
  return Napi::String::New(this->Env(), "[ WebGLBuffer " + std::to_string(this->value_) + " ]");
}

Napi::Value WebGLBuffer::GetValue(Napi::CallbackInfo const& info) {
  return Napi::Number::New(this->Env(), this->value_);
}

Napi::FunctionReference WebGLContextEvent::constructor;

WebGLContextEvent::WebGLContextEvent(Napi::CallbackInfo const& info)
  : Napi::ObjectWrap<WebGLContextEvent>(info){};

Napi::Object WebGLContextEvent::New(GLuint value) {
  auto obj                               = WebGLContextEvent::constructor.New({});
  WebGLContextEvent::Unwrap(obj)->value_ = value;
  return obj;
};

Napi::Object WebGLContextEvent::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor =
    DefineClass(env,
                "WebGLContextEvent",
                {
                  InstanceAccessor("_", &WebGLContextEvent::GetValue, nullptr, napi_enumerable),
                  InstanceMethod("toString", &WebGLContextEvent::ToString),
                });
  WebGLContextEvent::constructor = Napi::Persistent(ctor);
  WebGLContextEvent::constructor.SuppressDestruct();
  // exports.Set("WebGLContextEvent", ctor);
  return exports;
};

Napi::Value WebGLContextEvent::ToString(Napi::CallbackInfo const& info) {
  return Napi::String::New(this->Env(),
                           "[ WebGLContextEvent " + std::to_string(this->value_) + " ]");
}

Napi::Value WebGLContextEvent::GetValue(Napi::CallbackInfo const& info) {
  return Napi::Number::New(this->Env(), this->value_);
}

Napi::FunctionReference WebGLFramebuffer::constructor;

WebGLFramebuffer::WebGLFramebuffer(Napi::CallbackInfo const& info)
  : Napi::ObjectWrap<WebGLFramebuffer>(info){};

Napi::Object WebGLFramebuffer::New(GLuint value) {
  auto obj                              = WebGLFramebuffer::constructor.New({});
  WebGLFramebuffer::Unwrap(obj)->value_ = value;
  return obj;
};

Napi::Object WebGLFramebuffer::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor =
    DefineClass(env,
                "WebGLFramebuffer",
                {
                  InstanceAccessor("_", &WebGLFramebuffer::GetValue, nullptr, napi_enumerable),
                  InstanceMethod("toString", &WebGLFramebuffer::ToString),
                });
  WebGLFramebuffer::constructor = Napi::Persistent(ctor);
  WebGLFramebuffer::constructor.SuppressDestruct();
  // exports.Set("WebGLFramebuffer", ctor);
  return exports;
};

Napi::Value WebGLFramebuffer::ToString(Napi::CallbackInfo const& info) {
  return Napi::String::New(this->Env(),
                           "[ WebGLFramebuffer " + std::to_string(this->value_) + " ]");
}

Napi::Value WebGLFramebuffer::GetValue(Napi::CallbackInfo const& info) {
  return Napi::Number::New(this->Env(), this->value_);
}

Napi::FunctionReference WebGLProgram::constructor;

WebGLProgram::WebGLProgram(Napi::CallbackInfo const& info) : Napi::ObjectWrap<WebGLProgram>(info){};

Napi::Object WebGLProgram::New(GLuint value) {
  auto obj                          = WebGLProgram::constructor.New({});
  WebGLProgram::Unwrap(obj)->value_ = value;
  return obj;
};

Napi::Object WebGLProgram::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor =
    DefineClass(env,
                "WebGLProgram",
                {
                  InstanceAccessor("_", &WebGLProgram::GetValue, nullptr, napi_enumerable),
                  InstanceMethod("toString", &WebGLProgram::ToString),
                });
  WebGLProgram::constructor = Napi::Persistent(ctor);
  WebGLProgram::constructor.SuppressDestruct();
  // exports.Set("WebGLProgram", ctor);
  return exports;
};

Napi::Value WebGLProgram::ToString(Napi::CallbackInfo const& info) {
  return Napi::String::New(this->Env(), "[ WebGLProgram " + std::to_string(this->value_) + " ]");
}

Napi::Value WebGLProgram::GetValue(Napi::CallbackInfo const& info) {
  return Napi::Number::New(this->Env(), this->value_);
}

Napi::FunctionReference WebGLQuery::constructor;

WebGLQuery::WebGLQuery(Napi::CallbackInfo const& info) : Napi::ObjectWrap<WebGLQuery>(info){};

Napi::Object WebGLQuery::New(GLuint value) {
  auto obj                        = WebGLQuery::constructor.New({});
  WebGLQuery::Unwrap(obj)->value_ = value;
  return obj;
};

Napi::Object WebGLQuery::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor =
    DefineClass(env,
                "WebGLQuery",
                {
                  InstanceAccessor("_", &WebGLQuery::GetValue, nullptr, napi_enumerable),
                  InstanceMethod("toString", &WebGLQuery::ToString),
                });
  WebGLQuery::constructor = Napi::Persistent(ctor);
  WebGLQuery::constructor.SuppressDestruct();
  // exports.Set("WebGLQuery", ctor);
  return exports;
};

Napi::Value WebGLQuery::ToString(Napi::CallbackInfo const& info) {
  return Napi::String::New(this->Env(), "[ WebGLQuery " + std::to_string(this->value_) + " ]");
}

Napi::Value WebGLQuery::GetValue(Napi::CallbackInfo const& info) {
  return Napi::Number::New(this->Env(), this->value_);
}

Napi::FunctionReference WebGLRenderbuffer::constructor;

WebGLRenderbuffer::WebGLRenderbuffer(Napi::CallbackInfo const& info)
  : Napi::ObjectWrap<WebGLRenderbuffer>(info){};

Napi::Object WebGLRenderbuffer::New(GLuint value) {
  auto obj                               = WebGLRenderbuffer::constructor.New({});
  WebGLRenderbuffer::Unwrap(obj)->value_ = value;
  return obj;
};

Napi::Object WebGLRenderbuffer::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor =
    DefineClass(env,
                "WebGLRenderbuffer",
                {
                  InstanceAccessor("_", &WebGLRenderbuffer::GetValue, nullptr, napi_enumerable),
                  InstanceMethod("toString", &WebGLRenderbuffer::ToString),
                });
  WebGLRenderbuffer::constructor = Napi::Persistent(ctor);
  WebGLRenderbuffer::constructor.SuppressDestruct();
  // exports.Set("WebGLRenderbuffer", ctor);
  return exports;
};

Napi::Value WebGLRenderbuffer::ToString(Napi::CallbackInfo const& info) {
  return Napi::String::New(this->Env(),
                           "[ WebGLRenderbuffer " + std::to_string(this->value_) + " ]");
}

Napi::Value WebGLRenderbuffer::GetValue(Napi::CallbackInfo const& info) {
  return Napi::Number::New(this->Env(), this->value_);
}

Napi::FunctionReference WebGLSampler::constructor;

WebGLSampler::WebGLSampler(Napi::CallbackInfo const& info) : Napi::ObjectWrap<WebGLSampler>(info){};

Napi::Object WebGLSampler::New(GLuint value) {
  auto obj                          = WebGLSampler::constructor.New({});
  WebGLSampler::Unwrap(obj)->value_ = value;
  return obj;
};

Napi::Object WebGLSampler::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor =
    DefineClass(env,
                "WebGLSampler",
                {
                  InstanceAccessor("_", &WebGLSampler::GetValue, nullptr, napi_enumerable),
                  InstanceMethod("toString", &WebGLSampler::ToString),
                });
  WebGLSampler::constructor = Napi::Persistent(ctor);
  WebGLSampler::constructor.SuppressDestruct();
  // exports.Set("WebGLSampler", ctor);
  return exports;
};

Napi::Value WebGLSampler::ToString(Napi::CallbackInfo const& info) {
  return Napi::String::New(this->Env(), "[ WebGLSampler " + std::to_string(this->value_) + " ]");
}

Napi::Value WebGLSampler::GetValue(Napi::CallbackInfo const& info) {
  return Napi::Number::New(this->Env(), this->value_);
}

Napi::FunctionReference WebGLShader::constructor;

WebGLShader::WebGLShader(Napi::CallbackInfo const& info) : Napi::ObjectWrap<WebGLShader>(info){};

Napi::Object WebGLShader::New(GLuint value) {
  auto obj                         = WebGLShader::constructor.New({});
  WebGLShader::Unwrap(obj)->value_ = value;
  return obj;
};

Napi::Object WebGLShader::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor =
    DefineClass(env,
                "WebGLShader",
                {
                  InstanceAccessor("_", &WebGLShader::GetValue, nullptr, napi_enumerable),
                  InstanceMethod("toString", &WebGLShader::ToString),
                });
  WebGLShader::constructor = Napi::Persistent(ctor);
  WebGLShader::constructor.SuppressDestruct();
  // exports.Set("WebGLShader", ctor);
  return exports;
};

Napi::Value WebGLShader::ToString(Napi::CallbackInfo const& info) {
  return Napi::String::New(this->Env(), "[ WebGLShader " + std::to_string(this->value_) + " ]");
}

Napi::Value WebGLShader::GetValue(Napi::CallbackInfo const& info) {
  return Napi::Number::New(this->Env(), this->value_);
}

Napi::FunctionReference WebGLSync::constructor;

WebGLSync::WebGLSync(Napi::CallbackInfo const& info) : Napi::ObjectWrap<WebGLSync>(info){};

Napi::Object WebGLSync::New(GLuint value) {
  auto obj                       = WebGLSync::constructor.New({});
  WebGLSync::Unwrap(obj)->value_ = value;
  return obj;
};

Napi::Object WebGLSync::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor =
    DefineClass(env,
                "WebGLSync",
                {
                  InstanceAccessor("_", &WebGLSync::GetValue, nullptr, napi_enumerable),
                  InstanceMethod("toString", &WebGLSync::ToString),
                });
  WebGLSync::constructor = Napi::Persistent(ctor);
  WebGLSync::constructor.SuppressDestruct();
  // exports.Set("WebGLSync", ctor);
  return exports;
};

Napi::Value WebGLSync::ToString(Napi::CallbackInfo const& info) {
  return Napi::String::New(this->Env(), "[ WebGLSync " + std::to_string(this->value_) + " ]");
}

Napi::Value WebGLSync::GetValue(Napi::CallbackInfo const& info) {
  return Napi::Number::New(this->Env(), this->value_);
}

Napi::FunctionReference WebGLTexture::constructor;

WebGLTexture::WebGLTexture(Napi::CallbackInfo const& info) : Napi::ObjectWrap<WebGLTexture>(info){};

Napi::Object WebGLTexture::New(GLuint value) {
  auto obj                          = WebGLTexture::constructor.New({});
  WebGLTexture::Unwrap(obj)->value_ = value;
  return obj;
};

Napi::Object WebGLTexture::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor =
    DefineClass(env,
                "WebGLTexture",
                {
                  InstanceAccessor("_", &WebGLTexture::GetValue, nullptr, napi_enumerable),
                  InstanceMethod("toString", &WebGLTexture::ToString),
                });
  WebGLTexture::constructor = Napi::Persistent(ctor);
  WebGLTexture::constructor.SuppressDestruct();
  // exports.Set("WebGLTexture", ctor);
  return exports;
};

Napi::Value WebGLTexture::ToString(Napi::CallbackInfo const& info) {
  return Napi::String::New(this->Env(), "[ WebGLTexture " + std::to_string(this->value_) + " ]");
}

Napi::Value WebGLTexture::GetValue(Napi::CallbackInfo const& info) {
  return Napi::Number::New(this->Env(), this->value_);
}

Napi::FunctionReference WebGLTransformFeedback::constructor;

WebGLTransformFeedback::WebGLTransformFeedback(Napi::CallbackInfo const& info)
  : Napi::ObjectWrap<WebGLTransformFeedback>(info){};

Napi::Object WebGLTransformFeedback::New(GLuint value) {
  auto obj                                    = WebGLTransformFeedback::constructor.New({});
  WebGLTransformFeedback::Unwrap(obj)->value_ = value;
  return obj;
};

Napi::Object WebGLTransformFeedback::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor = DefineClass(
    env,
    "WebGLTransformFeedback",
    {
      InstanceAccessor("_", &WebGLTransformFeedback::GetValue, nullptr, napi_enumerable),
      InstanceMethod("toString", &WebGLTransformFeedback::ToString),
    });
  WebGLTransformFeedback::constructor = Napi::Persistent(ctor);
  WebGLTransformFeedback::constructor.SuppressDestruct();
  // exports.Set("WebGLTransformFeedback", ctor);
  return exports;
};

Napi::Value WebGLTransformFeedback::ToString(Napi::CallbackInfo const& info) {
  return Napi::String::New(this->Env(),
                           "[ WebGLTransformFeedback " + std::to_string(this->value_) + " ]");
}

Napi::Value WebGLTransformFeedback::GetValue(Napi::CallbackInfo const& info) {
  return Napi::Number::New(this->Env(), this->value_);
}

Napi::FunctionReference WebGLUniformLocation::constructor;

WebGLUniformLocation::WebGLUniformLocation(Napi::CallbackInfo const& info)
  : Napi::ObjectWrap<WebGLUniformLocation>(info){};

Napi::Object WebGLUniformLocation::New(GLuint value) {
  auto obj                                  = WebGLUniformLocation::constructor.New({});
  WebGLUniformLocation::Unwrap(obj)->value_ = value;
  return obj;
};

Napi::Object WebGLUniformLocation::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor =
    DefineClass(env,
                "WebGLUniformLocation",
                {
                  InstanceAccessor("_", &WebGLUniformLocation::GetValue, nullptr, napi_enumerable),
                  InstanceMethod("toString", &WebGLUniformLocation::ToString),
                });
  WebGLUniformLocation::constructor = Napi::Persistent(ctor);
  WebGLUniformLocation::constructor.SuppressDestruct();
  // exports.Set("WebGLUniformLocation", ctor);
  return exports;
};

Napi::Value WebGLUniformLocation::ToString(Napi::CallbackInfo const& info) {
  return Napi::String::New(this->Env(),
                           "[ WebGLUniformLocation " + std::to_string(this->value_) + " ]");
}

Napi::Value WebGLUniformLocation::GetValue(Napi::CallbackInfo const& info) {
  return Napi::Number::New(this->Env(), this->value_);
}

Napi::FunctionReference WebGLVertexArrayObject::constructor;

WebGLVertexArrayObject::WebGLVertexArrayObject(Napi::CallbackInfo const& info)
  : Napi::ObjectWrap<WebGLVertexArrayObject>(info){};

Napi::Object WebGLVertexArrayObject::New(GLuint value) {
  auto obj                                    = WebGLVertexArrayObject::constructor.New({});
  WebGLVertexArrayObject::Unwrap(obj)->value_ = value;
  return obj;
};

Napi::Object WebGLVertexArrayObject::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor = DefineClass(
    env,
    "WebGLVertexArrayObject",
    {
      InstanceAccessor("_", &WebGLVertexArrayObject::GetValue, nullptr, napi_enumerable),
      InstanceMethod("toString", &WebGLVertexArrayObject::ToString),
    });
  WebGLVertexArrayObject::constructor = Napi::Persistent(ctor);
  WebGLVertexArrayObject::constructor.SuppressDestruct();
  // exports.Set("WebGLVertexArrayObject", ctor);
  return exports;
};

Napi::Value WebGLVertexArrayObject::ToString(Napi::CallbackInfo const& info) {
  return Napi::String::New(this->Env(),
                           "[ WebGLVertexArrayObject " + std::to_string(this->value_) + " ]");
}

Napi::Value WebGLVertexArrayObject::GetValue(Napi::CallbackInfo const& info) {
  return Napi::Number::New(this->Env(), this->value_);
}

}  // namespace node_webgl
