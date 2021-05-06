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

// GL_EXPORT GLuint glCreateProgram (void);
Napi::Value WebGL2RenderingContext::CreateProgram(Napi::CallbackInfo const& info) {
  return WebGLProgram::New(info.Env(), GL_EXPORT::glCreateProgram());
}

// GL_EXPORT void glDeleteProgram (GLuint program);
void WebGL2RenderingContext::DeleteProgram(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glDeleteProgram(args[0]);
}

// GL_EXPORT void glGetProgramInfoLog (GLuint program, GLsizei bufSize, GLsizei* length, GLchar*
// infoLog);
Napi::Value WebGL2RenderingContext::GetProgramInfoLog(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint program     = args[0];
  GLint max_length{};
  GL_EXPORT::glGetProgramiv(program, GL_INFO_LOG_LENGTH, &max_length);
  if (max_length > 0) {
    GLint length{};
    GLchar* info_log = reinterpret_cast<GLchar*>(std::malloc(max_length));
    GL_EXPORT::glGetProgramInfoLog(program, max_length, &length, info_log);
    return CPPToNapi(info)(std::string{info_log, static_cast<size_t>(length)});
  }
  return info.Env().Null();
}

// GL_EXPORT void glGetProgramiv (GLuint program, GLenum pname, GLint* param);
Napi::Value WebGL2RenderingContext::GetProgramParameter(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLuint program    = args[0];
  GLuint pname      = args[1];
  switch (pname) {
    case GL_LINK_STATUS:
    case GL_DELETE_STATUS:
    case GL_VALIDATE_STATUS: {
      GLint param{};
      GL_EXPORT::glGetProgramiv(program, pname, &param);
      return CPPToNapi(info)(static_cast<bool>(param));
    }
    case GL_ACTIVE_UNIFORMS:
    case GL_ATTACHED_SHADERS:
    case GL_ACTIVE_ATTRIBUTES:
    case GL_ACTIVE_UNIFORM_BLOCKS:
    case GL_TRANSFORM_FEEDBACK_VARYINGS:
    case GL_TRANSFORM_FEEDBACK_BUFFER_MODE: {
      GLint param{};
      GL_EXPORT::glGetProgramiv(program, pname, &param);
      return CPPToNapi(info)(param);
    }
    default: GLEW_THROW(info.Env(), GL_INVALID_ENUM);
  }
}

// GL_EXPORT GLboolean glIsProgram (GLuint program);
Napi::Value WebGL2RenderingContext::IsProgram(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  auto is_program   = GL_EXPORT::glIsProgram(args[0]);
  return CPPToNapi(info.Env())(is_program);
}

// GL_EXPORT void glLinkProgram (GLuint program);
void WebGL2RenderingContext::LinkProgram(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glLinkProgram(args[0]);
}

// GL_EXPORT void glUseProgram (GLuint program);
void WebGL2RenderingContext::UseProgram(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glUseProgram(args[0]);
}

// GL_EXPORT void glValidateProgram (GLuint program);
void WebGL2RenderingContext::ValidateProgram(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glValidateProgram(args[0]);
}

}  // namespace nv
