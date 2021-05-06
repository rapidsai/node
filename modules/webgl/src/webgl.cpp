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

#include "webgl.hpp"

namespace nv {

WebGLActiveInfo::WebGLActiveInfo(Napi::CallbackInfo const& info)
  : EnvLocalObjectWrap<WebGLActiveInfo>(info) {
  if (info.Length() > 0) size_ = info[0].ToNumber();
  if (info.Length() > 1) type_ = info[1].ToNumber();
  if (info.Length() > 2) name_ = info[2].ToString();
};

Napi::Function WebGLActiveInfo::Init(Napi::Env const& env, Napi::Object exports) {
  return DefineClass(
    env,
    "WebGLActiveInfo",
    {
      InstanceAccessor("size", &WebGLActiveInfo::GetSize, nullptr, napi_enumerable),
      InstanceAccessor("type", &WebGLActiveInfo::GetType, nullptr, napi_enumerable),
      InstanceAccessor("name", &WebGLActiveInfo::GetName, nullptr, napi_enumerable),
      InstanceMethod("toString", &WebGLActiveInfo::ToString),
    });
};

WebGLActiveInfo::wrapper_t WebGLActiveInfo::New(Napi::Env const& env,
                                                GLint size,
                                                GLuint type,
                                                std::string name) {
  return EnvLocalObjectWrap<WebGLActiveInfo>::New(env, size, type, name);
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

WebGLShaderPrecisionFormat::WebGLShaderPrecisionFormat(Napi::CallbackInfo const& info)
  : EnvLocalObjectWrap<WebGLShaderPrecisionFormat>(info) {
  if (info.Length() > 0) rangeMin_ = info[0].ToNumber();
  if (info.Length() > 1) rangeMax_ = info[1].ToNumber();
  if (info.Length() > 2) precision_ = info[2].ToNumber();
};

Napi::Function WebGLShaderPrecisionFormat::Init(Napi::Env const& env, Napi::Object exports) {
  return DefineClass(
    env,
    "WebGLShaderPrecisionFormat",
    {
      InstanceAccessor(
        "rangeMin", &WebGLShaderPrecisionFormat::GetRangeMax, nullptr, napi_enumerable),
      InstanceAccessor(
        "rangeMax", &WebGLShaderPrecisionFormat::GetRangeMin, nullptr, napi_enumerable),
      InstanceAccessor(
        "precision", &WebGLShaderPrecisionFormat::GetPrecision, nullptr, napi_enumerable),
      InstanceMethod("toString", &WebGLShaderPrecisionFormat::ToString),
    });
};

WebGLShaderPrecisionFormat::wrapper_t WebGLShaderPrecisionFormat::New(Napi::Env const& env,
                                                                      GLint rangeMin,
                                                                      GLint rangeMax,
                                                                      GLint precision) {
  return EnvLocalObjectWrap<WebGLShaderPrecisionFormat>::New(env, rangeMin, rangeMax, precision);
};

Napi::Value WebGLShaderPrecisionFormat::GetRangeMin(Napi::CallbackInfo const& info) {
  return Napi::Number::New(this->Env(), this->rangeMin_);
}

Napi::Value WebGLShaderPrecisionFormat::GetRangeMax(Napi::CallbackInfo const& info) {
  return Napi::Number::New(this->Env(), this->rangeMax_);
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

WebGLBuffer::WebGLBuffer(Napi::CallbackInfo const& info) : EnvLocalObjectWrap<WebGLBuffer>(info) {
  if (info.Length() > 0) value_ = info[0].ToNumber();
};

WebGLBuffer::wrapper_t WebGLBuffer::New(Napi::Env const& env, GLuint value) {
  return EnvLocalObjectWrap<WebGLBuffer>::New(env, value);
};

Napi::Function WebGLBuffer::Init(Napi::Env const& env, Napi::Object exports) {
  return DefineClass(env,
                     "WebGLBuffer",
                     {
                       InstanceAccessor("ptr", &WebGLBuffer::GetValue, nullptr, napi_enumerable),
                       InstanceMethod("toString", &WebGLBuffer::ToString),
                     });
};

Napi::Value WebGLBuffer::ToString(Napi::CallbackInfo const& info) {
  return Napi::String::New(this->Env(), "[ WebGLBuffer " + std::to_string(this->value_) + " ]");
}

Napi::Value WebGLBuffer::GetValue(Napi::CallbackInfo const& info) {
  return Napi::Number::New(this->Env(), this->value_);
}

WebGLContextEvent::WebGLContextEvent(Napi::CallbackInfo const& info)
  : EnvLocalObjectWrap<WebGLContextEvent>(info) {
  if (info.Length() > 0) value_ = info[0].ToNumber();
};

WebGLContextEvent::wrapper_t WebGLContextEvent::New(Napi::Env const& env, GLuint value) {
  return EnvLocalObjectWrap<WebGLContextEvent>::New(env, value);
};

Napi::Function WebGLContextEvent::Init(Napi::Env const& env, Napi::Object exports) {
  return DefineClass(
    env,
    "WebGLContextEvent",
    {
      InstanceAccessor("ptr", &WebGLContextEvent::GetValue, nullptr, napi_enumerable),
      InstanceMethod("toString", &WebGLContextEvent::ToString),
    });
};

Napi::Value WebGLContextEvent::ToString(Napi::CallbackInfo const& info) {
  return Napi::String::New(this->Env(),
                           "[ WebGLContextEvent " + std::to_string(this->value_) + " ]");
}

Napi::Value WebGLContextEvent::GetValue(Napi::CallbackInfo const& info) {
  return Napi::Number::New(this->Env(), this->value_);
}

WebGLFramebuffer::WebGLFramebuffer(Napi::CallbackInfo const& info)
  : EnvLocalObjectWrap<WebGLFramebuffer>(info) {
  if (info.Length() > 0) value_ = info[0].ToNumber();
};

WebGLFramebuffer::wrapper_t WebGLFramebuffer::New(Napi::Env const& env, GLuint value) {
  return EnvLocalObjectWrap<WebGLFramebuffer>::New(env, value);
};

Napi::Function WebGLFramebuffer::Init(Napi::Env const& env, Napi::Object exports) {
  return DefineClass(
    env,
    "WebGLFramebuffer",
    {
      InstanceAccessor("ptr", &WebGLFramebuffer::GetValue, nullptr, napi_enumerable),
      InstanceMethod("toString", &WebGLFramebuffer::ToString),
    });
};

Napi::Value WebGLFramebuffer::ToString(Napi::CallbackInfo const& info) {
  return Napi::String::New(this->Env(),
                           "[ WebGLFramebuffer " + std::to_string(this->value_) + " ]");
}

Napi::Value WebGLFramebuffer::GetValue(Napi::CallbackInfo const& info) {
  return Napi::Number::New(this->Env(), this->value_);
}

WebGLProgram::WebGLProgram(Napi::CallbackInfo const& info)
  : EnvLocalObjectWrap<WebGLProgram>(info) {
  if (info.Length() > 0) value_ = info[0].ToNumber();
};

WebGLProgram::wrapper_t WebGLProgram::New(Napi::Env const& env, GLuint value) {
  return EnvLocalObjectWrap<WebGLProgram>::New(env, value);
};

Napi::Function WebGLProgram::Init(Napi::Env const& env, Napi::Object exports) {
  return DefineClass(env,
                     "WebGLProgram",
                     {
                       InstanceAccessor("ptr", &WebGLProgram::GetValue, nullptr, napi_enumerable),
                       InstanceMethod("toString", &WebGLProgram::ToString),
                     });
};

Napi::Value WebGLProgram::ToString(Napi::CallbackInfo const& info) {
  return Napi::String::New(this->Env(), "[ WebGLProgram " + std::to_string(this->value_) + " ]");
}

Napi::Value WebGLProgram::GetValue(Napi::CallbackInfo const& info) {
  return Napi::Number::New(this->Env(), this->value_);
}

WebGLQuery::WebGLQuery(Napi::CallbackInfo const& info) : EnvLocalObjectWrap<WebGLQuery>(info) {
  if (info.Length() > 0) value_ = info[0].ToNumber();
};

WebGLQuery::wrapper_t WebGLQuery::New(Napi::Env const& env, GLuint value) {
  return EnvLocalObjectWrap<WebGLQuery>::New(env, value);
};

Napi::Function WebGLQuery::Init(Napi::Env const& env, Napi::Object exports) {
  return DefineClass(env,
                     "WebGLQuery",
                     {
                       InstanceAccessor("ptr", &WebGLQuery::GetValue, nullptr, napi_enumerable),
                       InstanceMethod("toString", &WebGLQuery::ToString),
                     });
};

Napi::Value WebGLQuery::ToString(Napi::CallbackInfo const& info) {
  return Napi::String::New(this->Env(), "[ WebGLQuery " + std::to_string(this->value_) + " ]");
}

Napi::Value WebGLQuery::GetValue(Napi::CallbackInfo const& info) {
  return Napi::Number::New(this->Env(), this->value_);
}

WebGLRenderbuffer::WebGLRenderbuffer(Napi::CallbackInfo const& info)
  : EnvLocalObjectWrap<WebGLRenderbuffer>(info) {
  if (info.Length() > 0) value_ = info[0].ToNumber();
};

WebGLRenderbuffer::wrapper_t WebGLRenderbuffer::New(Napi::Env const& env, GLuint value) {
  return EnvLocalObjectWrap<WebGLRenderbuffer>::New(env, value);
};

Napi::Function WebGLRenderbuffer::Init(Napi::Env const& env, Napi::Object exports) {
  return DefineClass(
    env,
    "WebGLRenderbuffer",
    {
      InstanceAccessor("ptr", &WebGLRenderbuffer::GetValue, nullptr, napi_enumerable),
      InstanceMethod("toString", &WebGLRenderbuffer::ToString),
    });
};

Napi::Value WebGLRenderbuffer::ToString(Napi::CallbackInfo const& info) {
  return Napi::String::New(this->Env(),
                           "[ WebGLRenderbuffer " + std::to_string(this->value_) + " ]");
}

Napi::Value WebGLRenderbuffer::GetValue(Napi::CallbackInfo const& info) {
  return Napi::Number::New(this->Env(), this->value_);
}

WebGLSampler::WebGLSampler(Napi::CallbackInfo const& info)
  : EnvLocalObjectWrap<WebGLSampler>(info) {
  if (info.Length() > 0) value_ = info[0].ToNumber();
};

WebGLSampler::wrapper_t WebGLSampler::New(Napi::Env const& env, GLuint value) {
  return EnvLocalObjectWrap<WebGLSampler>::New(env, value);
};

Napi::Function WebGLSampler::Init(Napi::Env const& env, Napi::Object exports) {
  return DefineClass(env,
                     "WebGLSampler",
                     {
                       InstanceAccessor("ptr", &WebGLSampler::GetValue, nullptr, napi_enumerable),
                       InstanceMethod("toString", &WebGLSampler::ToString),
                     });
};

Napi::Value WebGLSampler::ToString(Napi::CallbackInfo const& info) {
  return Napi::String::New(this->Env(), "[ WebGLSampler " + std::to_string(this->value_) + " ]");
}

Napi::Value WebGLSampler::GetValue(Napi::CallbackInfo const& info) {
  return Napi::Number::New(this->Env(), this->value_);
}

WebGLShader::WebGLShader(Napi::CallbackInfo const& info) : EnvLocalObjectWrap<WebGLShader>(info) {
  if (info.Length() > 0) value_ = info[0].ToNumber();
};

WebGLShader::wrapper_t WebGLShader::New(Napi::Env const& env, GLuint value) {
  return EnvLocalObjectWrap<WebGLShader>::New(env, value);
};

Napi::Function WebGLShader::Init(Napi::Env const& env, Napi::Object exports) {
  return DefineClass(env,
                     "WebGLShader",
                     {
                       InstanceAccessor("ptr", &WebGLShader::GetValue, nullptr, napi_enumerable),
                       InstanceMethod("toString", &WebGLShader::ToString),
                     });
};

Napi::Value WebGLShader::ToString(Napi::CallbackInfo const& info) {
  return Napi::String::New(this->Env(), "[ WebGLShader " + std::to_string(this->value_) + " ]");
}

Napi::Value WebGLShader::GetValue(Napi::CallbackInfo const& info) {
  return Napi::Number::New(this->Env(), this->value_);
}

WebGLSync::WebGLSync(Napi::CallbackInfo const& info) : EnvLocalObjectWrap<WebGLSync>(info) {
  if (info.Length() > 0) {
    value_ = reinterpret_cast<GLsync>(info[0].As<Napi::External<void>>().Data());
  }
};

WebGLSync::wrapper_t WebGLSync::New(Napi::Env const& env, GLsync value) {
  return EnvLocalObjectWrap<WebGLSync>::New(env, {Napi::External<void>::New(env, value)});
};

Napi::Function WebGLSync::Init(Napi::Env const& env, Napi::Object exports) {
  return DefineClass(env,
                     "WebGLSync",
                     {
                       InstanceAccessor("ptr", &WebGLSync::GetValue, nullptr, napi_enumerable),
                       InstanceMethod("toString", &WebGLSync::ToString),
                     });
};

Napi::Value WebGLSync::ToString(Napi::CallbackInfo const& info) {
  return Napi::String::New(
    this->Env(), "[ WebGLSync " + std::to_string(reinterpret_cast<uintptr_t>(this->value_)) + " ]");
}

Napi::Value WebGLSync::GetValue(Napi::CallbackInfo const& info) {
  return Napi::Number::New(this->Env(), reinterpret_cast<uintptr_t>(this->value_));
}

WebGLTexture::WebGLTexture(Napi::CallbackInfo const& info)
  : EnvLocalObjectWrap<WebGLTexture>(info) {
  if (info.Length() > 0) value_ = info[0].ToNumber();
};

WebGLTexture::wrapper_t WebGLTexture::New(Napi::Env const& env, GLuint value) {
  return EnvLocalObjectWrap<WebGLTexture>::New(env, value);
};

Napi::Function WebGLTexture::Init(Napi::Env const& env, Napi::Object exports) {
  return DefineClass(env,
                     "WebGLTexture",
                     {
                       InstanceAccessor("ptr", &WebGLTexture::GetValue, nullptr, napi_enumerable),
                       InstanceMethod("toString", &WebGLTexture::ToString),
                     });
};

Napi::Value WebGLTexture::ToString(Napi::CallbackInfo const& info) {
  return Napi::String::New(this->Env(), "[ WebGLTexture " + std::to_string(this->value_) + " ]");
}

Napi::Value WebGLTexture::GetValue(Napi::CallbackInfo const& info) {
  return Napi::Number::New(this->Env(), this->value_);
}

WebGLTransformFeedback::WebGLTransformFeedback(Napi::CallbackInfo const& info)
  : EnvLocalObjectWrap<WebGLTransformFeedback>(info) {
  if (info.Length() > 0) value_ = info[0].ToNumber();
};

WebGLTransformFeedback::wrapper_t WebGLTransformFeedback::New(Napi::Env const& env, GLuint value) {
  return EnvLocalObjectWrap<WebGLTransformFeedback>::New(env, value);
};

Napi::Function WebGLTransformFeedback::Init(Napi::Env const& env, Napi::Object exports) {
  return DefineClass(
    env,
    "WebGLTransformFeedback",
    {
      InstanceAccessor("ptr", &WebGLTransformFeedback::GetValue, nullptr, napi_enumerable),
      InstanceMethod("toString", &WebGLTransformFeedback::ToString),
    });
};

Napi::Value WebGLTransformFeedback::ToString(Napi::CallbackInfo const& info) {
  return Napi::String::New(this->Env(),
                           "[ WebGLTransformFeedback " + std::to_string(this->value_) + " ]");
}

Napi::Value WebGLTransformFeedback::GetValue(Napi::CallbackInfo const& info) {
  return Napi::Number::New(this->Env(), this->value_);
}

WebGLUniformLocation::WebGLUniformLocation(Napi::CallbackInfo const& info)
  : EnvLocalObjectWrap<WebGLUniformLocation>(info) {
  if (info.Length() > 0) value_ = info[0].ToNumber();
};

WebGLUniformLocation::wrapper_t WebGLUniformLocation::New(Napi::Env const& env, GLint value) {
  return EnvLocalObjectWrap<WebGLUniformLocation>::New(env, value);
};

Napi::Function WebGLUniformLocation::Init(Napi::Env const& env, Napi::Object exports) {
  return DefineClass(
    env,
    "WebGLUniformLocation",
    {
      InstanceAccessor("ptr", &WebGLUniformLocation::GetValue, nullptr, napi_enumerable),
      InstanceMethod("toString", &WebGLUniformLocation::ToString),
    });
};

Napi::Value WebGLUniformLocation::ToString(Napi::CallbackInfo const& info) {
  return Napi::String::New(this->Env(),
                           "[ WebGLUniformLocation " + std::to_string(this->value_) + " ]");
}

Napi::Value WebGLUniformLocation::GetValue(Napi::CallbackInfo const& info) {
  return Napi::Number::New(this->Env(), this->value_);
}

WebGLVertexArrayObject::WebGLVertexArrayObject(Napi::CallbackInfo const& info)
  : EnvLocalObjectWrap<WebGLVertexArrayObject>(info) {
  if (info.Length() > 0) value_ = info[0].ToNumber();
};

WebGLVertexArrayObject::wrapper_t WebGLVertexArrayObject::New(Napi::Env const& env, GLuint value) {
  return EnvLocalObjectWrap<WebGLVertexArrayObject>::New(env, value);
};

Napi::Function WebGLVertexArrayObject::Init(Napi::Env const& env, Napi::Object exports) {
  return DefineClass(
    env,
    "WebGLVertexArrayObject",
    {
      InstanceAccessor("ptr", &WebGLVertexArrayObject::GetValue, nullptr, napi_enumerable),
      InstanceMethod("toString", &WebGLVertexArrayObject::ToString),
    });
};

Napi::Value WebGLVertexArrayObject::ToString(Napi::CallbackInfo const& info) {
  return Napi::String::New(this->Env(),
                           "[ WebGLVertexArrayObject " + std::to_string(this->value_) + " ]");
}

Napi::Value WebGLVertexArrayObject::GetValue(Napi::CallbackInfo const& info) {
  return Napi::Number::New(this->Env(), this->value_);
}

}  // namespace nv
