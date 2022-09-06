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

// GL_EXPORT void glAttachShader (GLuint program, GLuint shader);
void WebGL2RenderingContext::AttachShader(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glAttachShader(args[0], args[1]);
}

// GL_EXPORT void glCompileShader (GLuint shader);
void WebGL2RenderingContext::CompileShader(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glCompileShader(args[0]);
}

// GL_EXPORT GLuint glCreateShader (GLenum type);
Napi::Value WebGL2RenderingContext::CreateShader(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  auto shader       = GL_EXPORT::glCreateShader(args[0]);
  return WebGLShader::New(info.Env(), shader);
}

// GL_EXPORT void glDeleteShader (GLuint shader);
void WebGL2RenderingContext::DeleteShader(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glDeleteShader(args[0]);
}

// GL_EXPORT void glDetachShader (GLuint shader, GLuint shader);
void WebGL2RenderingContext::DetachShader(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glDetachShader(args[0], args[1]);
}

// GL_EXPORT void glGetAttachedShaders (GLuint shader, GLsizei maxCount, GLsizei* count, GLuint*
// shaders);
Napi::Value WebGL2RenderingContext::GetAttachedShaders(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLuint program    = args[0];
  GLint max_count{0};
  GL_EXPORT::glGetProgramiv(program, GL_ATTACHED_SHADERS, &max_count);
  if (max_count > 0) {
    GLint count{};
    std::vector<GLuint> shaders(max_count);
    GL_EXPORT::glGetAttachedShaders(program, max_count, &count, shaders.data());
    shaders.resize(count);
    shaders.shrink_to_fit();
    std::vector<Napi::Object> objs(count);
    std::transform(shaders.begin(), shaders.end(), objs.begin(), [&](GLuint s) {
      return WebGLShader::New(info.Env(), s);
    });
    return CPPToNapi(info)(objs);
  }
  return info.Env().Null();
}

// GL_EXPORT void glGetShaderInfoLog (GLuint shader, GLsizei bufSize, GLsizei* length, GLchar*
// infoLog);
Napi::Value WebGL2RenderingContext::GetShaderInfoLog(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLuint shader     = args[0];
  GLint max_length{0};
  GL_EXPORT::glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &max_length);
  if (max_length > 0) {
    GLint length{};
    GLchar* info_log = reinterpret_cast<GLchar*>(std::malloc(max_length));
    GL_EXPORT::glGetShaderInfoLog(shader, max_length, &length, info_log);
    return CPPToNapi(info)(std::string{info_log, static_cast<size_t>(length)});
  }
  return info.Env().Null();
}

// GL_EXPORT void glGetShaderiv (GLuint shader, GLenum pname, GLint* param);
Napi::Value WebGL2RenderingContext::GetShaderParameter(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLuint shader     = args[0];
  GLint pname       = args[1];
  // GLint param{};
  // GL_EXPORT::glGetShaderiv(shader, pname, &param);
  // return CPPToNapi(info)(param);
  switch (pname) {
    case GL_DELETE_STATUS:
    case GL_COMPILE_STATUS: {
      GLint param{};
      GL_EXPORT::glGetShaderiv(shader, pname, &param);
      return CPPToNapi(info)(static_cast<bool>(param));
    }
    case GL_SHADER_TYPE: {
      GLint param{};
      GL_EXPORT::glGetShaderiv(shader, pname, &param);
      return CPPToNapi(info)(param);
    }
    default: GLEW_THROW(info.Env(), GL_INVALID_ENUM);
  }
}

// GL_EXPORT void glGetShaderPrecisionFormat (GLenum shadertype, GLenum precisiontype, GLint* range,
// GLint *precision);
Napi::Value WebGL2RenderingContext::GetShaderPrecisionFormat(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  std::vector<GLint> range(2);
  GLint precision{};
  GL_EXPORT::glGetShaderPrecisionFormat(args[0], args[1], range.data(), &precision);
  return WebGLShaderPrecisionFormat::New(info.Env(), range[0], range[1], precision);
}

// GL_EXPORT void glGetShaderSource (GLuint obj, GLsizei maxLength, GLsizei* length, GLchar*
// source);
Napi::Value WebGL2RenderingContext::GetShaderSource(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLuint shader     = args[0];
  GLint max_length{};
  GL_EXPORT::glGetShaderiv(shader, GL_SHADER_SOURCE_LENGTH, &max_length);
  if (max_length > 0) {
    GLint length{};
    GLchar* shader_source = reinterpret_cast<GLchar*>(std::malloc(max_length));
    GL_EXPORT::glGetShaderSource(shader, max_length, &length, shader_source);
    return CPPToNapi(info)(std::string{shader_source, static_cast<size_t>(length)});
  }
  return info.Env().Null();
}

// GL_EXPORT void glGetTranslatedShaderSourceANGLE(GLuint shader, GLsizei bufsize, GLsizei* length,
// GLchar* source);
Napi::Value WebGL2RenderingContext::GetTranslatedShaderSource(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  if (GLEW_ANGLE_translated_shader_source) {
    GLuint shader = args[0];
    GLint max_length{};
    GL_EXPORT::glGetShaderiv(shader, GL_TRANSLATED_SHADER_SOURCE_LENGTH_ANGLE, &max_length);
    if (max_length > 0) {
      GLint length{};
      GLchar* shader_source = reinterpret_cast<GLchar*>(std::malloc(max_length));
      GL_EXPORT::glGetTranslatedShaderSourceANGLE(shader, max_length, &length, shader_source);
      return CPPToNapi(info)(std::string{shader_source, static_cast<size_t>(length)});
    }
  }
  return info.Env().Null();
}

// GL_EXPORT GLboolean glIsShader (GLuint shader);
Napi::Value WebGL2RenderingContext::IsShader(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  auto is_shader    = GL_EXPORT::glIsShader(args[0]);
  return CPPToNapi(info.Env())(is_shader);
}

// GL_EXPORT void glShaderSource (GLuint shader, GLsizei count, const GLchar *const* string, const
// GLint* length);
void WebGL2RenderingContext::ShaderSource(Napi::CallbackInfo const& info) {
  CallbackArgs args                = info;
  GLuint shader                    = args[0];
  std::vector<std::string> sources = args[1];
  std::vector<GLint> source_lengths(sources.size());
  std::vector<const GLchar*> source_ptrs(sources.size());
  auto idx = -1;
  std::for_each(sources.begin(), sources.end(), [&](std::string const& src) mutable {
    source_ptrs[++idx]  = src.data();
    source_lengths[idx] = src.size();
  });
  GL_EXPORT::glShaderSource(shader, sources.size(), source_ptrs.data(), source_lengths.data());
}

}  // namespace nv
