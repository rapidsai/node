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

// GL_EXPORT void glBindSampler (GLuint unit, GLuint sampler);
void WebGL2RenderingContext::BindSampler(Napi::CallbackInfo const& info) {
  CallbackArgs args            = info;
  std::vector<GLuint> samplers = args[0];
  GL_EXPORT::glBindSampler(args[0], args[1]);
}

// GL_EXPORT void glCreateSamplers (GLsizei n, GLuint* samplers);
Napi::Value WebGL2RenderingContext::CreateSampler(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLuint sampler{};
  GL_EXPORT::glCreateSamplers(1, &sampler);
  return WebGLSampler::New(info.Env(), sampler);
}

// GL_EXPORT void glCreateSamplers (GLsizei n, GLuint* samplers);
Napi::Value WebGL2RenderingContext::CreateSamplers(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  std::vector<GLuint> samplers(static_cast<size_t>(args[0]));
  GL_EXPORT::glCreateSamplers(samplers.size(), samplers.data());
  return CPPToNapi(info)(samplers);
}

// GL_EXPORT void glDeleteSamplers (GLsizei count, const GLuint * samplers);
void WebGL2RenderingContext::DeleteSampler(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLuint sampler    = args[0];
  GL_EXPORT::glDeleteSamplers(1, &sampler);
}

// GL_EXPORT void glDeleteSamplers (GLsizei count, const GLuint * samplers);
void WebGL2RenderingContext::DeleteSamplers(Napi::CallbackInfo const& info) {
  CallbackArgs args            = info;
  std::vector<GLuint> samplers = args[0];
  GL_EXPORT::glDeleteSamplers(samplers.size(), samplers.data());
}

// GL_EXPORT void glGetSamplerParameterfv (GLuint sampler, GLenum pname, GLfloat* params);
// GL_EXPORT void glGetSamplerParameteriv (GLuint sampler, GLenum pname, GLint* params);
Napi::Value WebGL2RenderingContext::GetSamplerParameter(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLuint sampler    = args[0];
  GLint pname       = args[1];
  switch (pname) {
    case GL_TEXTURE_MAX_LOD:
    case GL_TEXTURE_MIN_LOD: {
      GLfloat params{};
      GL_EXPORT::glGetSamplerParameterfv(sampler, pname, &params);
      return CPPToNapi(info)(params);
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
      return CPPToNapi(info)(params);
    }
    default: GLEW_THROW(info.Env(), GL_INVALID_ENUM);
  }
}

// GL_EXPORT GLboolean glIsSampler (GLuint sampler);
Napi::Value WebGL2RenderingContext::IsSampler(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  auto is_sampler   = GL_EXPORT::glIsSampler(args[0]);
  return CPPToNapi(info.Env())(is_sampler);
}

// GL_EXPORT void glSamplerParameterf (GLuint sampler, GLenum pname, GLfloat param);
void WebGL2RenderingContext::SamplerParameterf(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glSamplerParameterf(args[0], args[1], args[2]);
}

// GL_EXPORT void glSamplerParameteri (GLuint sampler, GLenum pname, GLint param);
void WebGL2RenderingContext::SamplerParameteri(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glSamplerParameteri(args[0], args[1], args[2]);
}

}  // namespace nv
