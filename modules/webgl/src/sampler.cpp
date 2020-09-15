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

// GL_EXPORT void glBindSampler (GLuint unit, GLuint sampler);
Napi::Value WebGL2RenderingContext::BindSampler(Napi::CallbackInfo const& info) {
  auto env                     = info.Env();
  std::vector<GLuint> samplers = FromJS(info[0]);
  GL_EXPORT::glBindSampler(FromJS(info[0]), FromJS(info[1]));
  return env.Undefined();
}

// GL_EXPORT void glCreateSamplers (GLsizei n, GLuint* samplers);
Napi::Value WebGL2RenderingContext::CreateSampler(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GLuint sampler{};
  GL_EXPORT::glCreateSamplers(1, &sampler);
  return WebGLSampler::New(sampler);
}

// GL_EXPORT void glCreateSamplers (GLsizei n, GLuint* samplers);
Napi::Value WebGL2RenderingContext::CreateSamplers(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  std::vector<GLuint> samplers(static_cast<size_t>(FromJS(info[0])));
  GL_EXPORT::glCreateSamplers(samplers.size(), samplers.data());
  return ToNapi(env)(samplers);
}

// GL_EXPORT void glDeleteSamplers (GLsizei count, const GLuint * samplers);
Napi::Value WebGL2RenderingContext::DeleteSampler(Napi::CallbackInfo const& info) {
  auto env       = info.Env();
  GLuint sampler = FromJS(info[0]);
  GL_EXPORT::glDeleteSamplers(1, &sampler);
  return env.Undefined();
}

// GL_EXPORT void glDeleteSamplers (GLsizei count, const GLuint * samplers);
Napi::Value WebGL2RenderingContext::DeleteSamplers(Napi::CallbackInfo const& info) {
  auto env                     = info.Env();
  std::vector<GLuint> samplers = FromJS(info[0]);
  GL_EXPORT::glDeleteSamplers(samplers.size(), samplers.data());
  return env.Undefined();
}

// GL_EXPORT void glGetSamplerParameterfv (GLuint sampler, GLenum pname, GLfloat* params);
// GL_EXPORT void glGetSamplerParameteriv (GLuint sampler, GLenum pname, GLint* params);
Napi::Value WebGL2RenderingContext::GetSamplerParameter(Napi::CallbackInfo const& info) {
  auto env       = info.Env();
  GLuint sampler = FromJS(info[0]);
  GLint pname    = FromJS(info[1]);
  switch (pname) {
    case GL_TEXTURE_MAX_LOD:
    case GL_TEXTURE_MIN_LOD: {
      GLfloat params{};
      GL_EXPORT::glGetSamplerParameterfv(sampler, pname, &params);
      return ToNapi(env)(params);
    }
    case GL_TEXTURE_WRAP_R:
    case GL_TEXTURE_WRAP_S:
    case GL_TEXTURE_WRAP_T:
    case GL_TEXTURE_MAG_FILTER:
    case GL_TEXTURE_MIN_FILTER:
    case GL_TEXTURE_COMPARE_FUNC:
    case GL_TEXTURE_COMPARE_MODE: {
      GLint params{};
      GL_EXPORT::glGetSamplerParameteriv(sampler, pname, &params);
      return ToNapi(env)(params);
    }
    default: GLEW_THROW(env, GL_INVALID_ENUM);
  }
}

// GL_EXPORT GLboolean glIsSampler (GLuint sampler);
Napi::Value WebGL2RenderingContext::IsSampler(Napi::CallbackInfo const& info) {
  return ToNapi(info.Env())(GL_EXPORT::glIsSampler(FromJS(info[0])));
}

// GL_EXPORT void glSamplerParameterf (GLuint sampler, GLenum pname, GLfloat param);
Napi::Value WebGL2RenderingContext::SamplerParameterf(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glSamplerParameterf(FromJS(info[0]), FromJS(info[1]), FromJS(info[2]));
  return env.Undefined();
}

// GL_EXPORT void glSamplerParameteri (GLuint sampler, GLenum pname, GLint param);
Napi::Value WebGL2RenderingContext::SamplerParameteri(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glSamplerParameteri(FromJS(info[0]), FromJS(info[1]), FromJS(info[2]));
  return env.Undefined();
}

}  // namespace node_webgl
