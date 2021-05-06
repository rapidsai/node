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

namespace detail {

// GL_EXPORT void glGetUniformfv (GLuint program, GLint location, GLfloat* params);
Napi::Value glGetUniformfv(Napi::Env const& env,
                           GLuint const program,
                           GLint const location,
                           GLuint const size) {
  if (size == 1) {
    GLfloat params{};
    GL_EXPORT::glGetUniformfv(program, location, &params);
    return CPPToNapi(env)(params);
  }
  auto buf = Napi::ArrayBuffer::New(env, size * sizeof(GLfloat));
  GL_EXPORT::glGetUniformfv(program, location, static_cast<GLfloat*>(buf.Data()));
  return Napi::Float32Array::New(env, size, buf, 0);
}

// GL_EXPORT void glGetUniformiv (GLuint program, GLint location, GLint* params);
Napi::Value glGetUniformiv(Napi::Env const& env,
                           GLuint const program,
                           GLint const location,
                           GLuint const size) {
  if (size == 1) {
    GLint params{};
    GL_EXPORT::glGetUniformiv(program, location, &params);
    return CPPToNapi(env)(params);
  }
  auto buf = Napi::ArrayBuffer::New(env, size * sizeof(GLint));
  GL_EXPORT::glGetUniformiv(program, location, static_cast<GLint*>(buf.Data()));
  return Napi::Int32Array::New(env, size, buf, 0);
}

// GL_EXPORT void glGetUniformuiv (GLuint program, GLint location, GLuint* params);
Napi::Value glGetUniformuiv(Napi::Env const& env,
                            GLuint const program,
                            GLint const location,
                            GLuint const size) {
  if (size == 1) {
    GLuint params{};
    GL_EXPORT::glGetUniformuiv(program, location, &params);
    return CPPToNapi(env)(params);
  }
  auto buf = Napi::ArrayBuffer::New(env, size * sizeof(GLuint));
  GL_EXPORT::glGetUniformuiv(program, location, static_cast<GLuint*>(buf.Data()));
  return Napi::Uint32Array::New(env, size, buf, 0);
}

// GL_EXPORT void glGetUniformdv (GLuint program, GLint location, GLdouble* params);
Napi::Value glGetUniformdv(Napi::Env const& env,
                           GLuint const program,
                           GLint const location,
                           GLuint const size) {
  if (size == 1) {
    GLdouble params{};
    GL_EXPORT::glGetUniformdv(program, location, &params);
    return CPPToNapi(env)(params);
  }
  auto buf = Napi::ArrayBuffer::New(env, size * sizeof(GLdouble));
  GL_EXPORT::glGetUniformdv(program, location, static_cast<GLdouble*>(buf.Data()));
  return Napi::Float64Array::New(env, size, buf, 0);
}

// GL_EXPORT void glGetUniformiv (GLuint program, GLint location, GLint* params);
Napi::Value glGetUniformbv(Napi::Env const& env,
                           GLuint const program,
                           GLint const location,
                           GLuint const size) {
  std::vector<bool> bools(size);
  std::vector<GLint> params(size);
  GL_EXPORT::glGetUniformiv(program, location, params.data());
  std::transform(
    params.begin(), params.end(), bools.begin(), [&](auto x) { return static_cast<bool>(x); });
  if (size == 1) {
    return CPPToNapi(env)(bools[0]);
  } else {
    return CPPToNapi(env)(bools);
  }
}

}  // namespace detail

// GL_EXPORT void glGetActiveUniform (GLuint program, GLuint index, GLsizei maxLength, GLsizei*
// length, GLint* size, GLenum* type, GLchar* name);
Napi::Value WebGL2RenderingContext::GetActiveUniform(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLuint program    = args[0];
  GLint max_length{};
  GL_EXPORT::glGetProgramiv(program, GL_ACTIVE_UNIFORM_MAX_LENGTH, &max_length);
  if (max_length > 0) {
    GLuint type{};
    GLint size{}, length{};
    GLchar* name = reinterpret_cast<GLchar*>(std::malloc(max_length));
    GL_EXPORT::glGetActiveUniform(program, args[1], max_length, &length, &size, &type, name);
    return CPPToNapi(info)(
      {"size", "type", "name"},
      std::make_tuple(size, type, std::string{name, static_cast<size_t>(length)}));
  }
  return info.Env().Null();
}

// GL_EXPORT void glGetUniformfv (GLuint program, GLint location, GLfloat* params);
// GL_EXPORT void glGetUniformiv (GLuint program, GLint location, GLint* params);
// GL_EXPORT void glGetUniformuiv (GLuint program, GLint location, GLuint* params);
// GL_EXPORT void glGetUniformdv (GLuint program, GLint location, GLdouble* params);
Napi::Value WebGL2RenderingContext::GetUniform(Napi::CallbackInfo const& info) {
  auto env          = info.Env();
  CallbackArgs args = info;
  GLuint program    = args[0];
  GLint location    = args[1];
  if (location < 0) { return info.Env().Null(); }
  GLint size{};
  GLuint type{};
  GL_EXPORT::glGetActiveUniform(program, location, 0, nullptr, &size, &type, nullptr);
  switch (type) {
    case GL_BOOL: return detail::glGetUniformbv(env, program, location, 1);
    case GL_BOOL_VEC2: return detail::glGetUniformbv(env, program, location, 2);
    case GL_BOOL_VEC3: return detail::glGetUniformbv(env, program, location, 3);
    case GL_BOOL_VEC4: return detail::glGetUniformbv(env, program, location, 4);

    case GL_FLOAT: return detail::glGetUniformfv(env, program, location, 1);
    case GL_FLOAT_VEC2: return detail::glGetUniformfv(env, program, location, 2);
    case GL_FLOAT_VEC3: return detail::glGetUniformfv(env, program, location, 3);
    case GL_FLOAT_VEC4: return detail::glGetUniformfv(env, program, location, 4);
    case GL_FLOAT_MAT2: return detail::glGetUniformfv(env, program, location, 2 * 2);
    case GL_FLOAT_MAT3: return detail::glGetUniformfv(env, program, location, 3 * 3);
    case GL_FLOAT_MAT4: return detail::glGetUniformfv(env, program, location, 4 * 4);
    case GL_FLOAT_MAT2x3: return detail::glGetUniformfv(env, program, location, 2 * 3);
    case GL_FLOAT_MAT2x4: return detail::glGetUniformfv(env, program, location, 2 * 4);
    case GL_FLOAT_MAT3x2: return detail::glGetUniformfv(env, program, location, 3 * 2);
    case GL_FLOAT_MAT3x4: return detail::glGetUniformfv(env, program, location, 3 * 4);
    case GL_FLOAT_MAT4x2: return detail::glGetUniformfv(env, program, location, 4 * 2);
    case GL_FLOAT_MAT4x3: return detail::glGetUniformfv(env, program, location, 4 * 3);

    case GL_INT: return detail::glGetUniformiv(env, program, location, 1);
    case GL_INT_VEC2: return detail::glGetUniformiv(env, program, location, 2);
    case GL_INT_VEC3: return detail::glGetUniformiv(env, program, location, 3);
    case GL_INT_VEC4: return detail::glGetUniformiv(env, program, location, 4);
    case GL_INT_SAMPLER_1D:
    case GL_INT_SAMPLER_2D:
    case GL_INT_SAMPLER_3D:
    case GL_INT_SAMPLER_CUBE:
    // case GL_INT_SAMPLER_1D_ARRAY:
    // case GL_INT_SAMPLER_2D_ARRAY:
    case GL_INT_SAMPLER_2D_MULTISAMPLE:
    // case GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
    case GL_INT_SAMPLER_BUFFER:
    case GL_INT_SAMPLER_2D_RECT:
    case GL_INT_IMAGE_1D:
    case GL_INT_IMAGE_2D:
    case GL_INT_IMAGE_3D:
    case GL_INT_IMAGE_2D_RECT:
    case GL_INT_IMAGE_CUBE:
    case GL_INT_IMAGE_BUFFER:
    // case GL_INT_IMAGE_1D_ARRAY:
    // case GL_INT_IMAGE_2D_ARRAY:
    case GL_INT_IMAGE_2D_MULTISAMPLE:
      // case GL_INT_IMAGE_2D_MULTISAMPLE_ARRAY:
      return detail::glGetUniformiv(env, program, location, 1);

    case GL_UNSIGNED_INT: return detail::glGetUniformuiv(env, program, location, 1);
    case GL_UNSIGNED_INT_VEC2: return detail::glGetUniformuiv(env, program, location, 2);
    case GL_UNSIGNED_INT_VEC3: return detail::glGetUniformuiv(env, program, location, 3);
    case GL_UNSIGNED_INT_VEC4: return detail::glGetUniformuiv(env, program, location, 4);
    case GL_UNSIGNED_INT_SAMPLER_1D:
    case GL_UNSIGNED_INT_SAMPLER_2D:
    case GL_UNSIGNED_INT_SAMPLER_3D:
    case GL_UNSIGNED_INT_SAMPLER_CUBE:
    // case GL_UNSIGNED_INT_SAMPLER_1D_ARRAY:
    // case GL_UNSIGNED_INT_SAMPLER_2D_ARRAY:
    case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE:
    // case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
    case GL_UNSIGNED_INT_SAMPLER_BUFFER:
    case GL_UNSIGNED_INT_SAMPLER_2D_RECT:
    case GL_UNSIGNED_INT_IMAGE_1D:
    case GL_UNSIGNED_INT_IMAGE_2D:
    case GL_UNSIGNED_INT_IMAGE_3D:
    case GL_UNSIGNED_INT_IMAGE_2D_RECT:
    case GL_UNSIGNED_INT_IMAGE_CUBE:
    case GL_UNSIGNED_INT_IMAGE_BUFFER:
    // case GL_UNSIGNED_INT_IMAGE_1D_ARRAY:
    // case GL_UNSIGNED_INT_IMAGE_2D_ARRAY:
    case GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE:
    // case GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE_ARRAY:
    case GL_UNSIGNED_INT_ATOMIC_COUNTER: return detail::glGetUniformuiv(env, program, location, 1);

    case GL_DOUBLE: return detail::glGetUniformdv(env, program, location, 1);
    case GL_DOUBLE_VEC2: return detail::glGetUniformdv(env, program, location, 2);
    case GL_DOUBLE_VEC3: return detail::glGetUniformdv(env, program, location, 3);
    case GL_DOUBLE_VEC4: return detail::glGetUniformdv(env, program, location, 4);
    case GL_DOUBLE_MAT2: return detail::glGetUniformdv(env, program, location, 2 * 2);
    case GL_DOUBLE_MAT3: return detail::glGetUniformdv(env, program, location, 3 * 3);
    case GL_DOUBLE_MAT4: return detail::glGetUniformdv(env, program, location, 4 * 4);
    case GL_DOUBLE_MAT2x3: return detail::glGetUniformdv(env, program, location, 2 * 3);
    case GL_DOUBLE_MAT2x4: return detail::glGetUniformdv(env, program, location, 2 * 4);
    case GL_DOUBLE_MAT3x2: return detail::glGetUniformdv(env, program, location, 3 * 2);
    case GL_DOUBLE_MAT3x4: return detail::glGetUniformdv(env, program, location, 3 * 4);
    case GL_DOUBLE_MAT4x2: return detail::glGetUniformdv(env, program, location, 4 * 2);
    case GL_DOUBLE_MAT4x3: return detail::glGetUniformdv(env, program, location, 4 * 3);

    case GL_SAMPLER_1D:
    case GL_SAMPLER_2D:
    case GL_SAMPLER_3D:
    case GL_SAMPLER_CUBE:
    case GL_SAMPLER_1D_SHADOW:
    case GL_SAMPLER_2D_SHADOW:
    // case GL_SAMPLER_1D_ARRAY:
    // case GL_SAMPLER_2D_ARRAY:
    // case GL_SAMPLER_1D_ARRAY_SHADOW:
    // case GL_SAMPLER_2D_ARRAY_SHADOW:
    case GL_SAMPLER_2D_MULTISAMPLE:
    // case GL_SAMPLER_2D_MULTISAMPLE_ARRAY:
    case GL_SAMPLER_CUBE_SHADOW:
    case GL_SAMPLER_BUFFER:
    case GL_SAMPLER_2D_RECT:
    case GL_SAMPLER_2D_RECT_SHADOW:

    case GL_IMAGE_1D:
    case GL_IMAGE_2D:
    case GL_IMAGE_3D:
    case GL_IMAGE_2D_RECT:
    case GL_IMAGE_CUBE:
    case GL_IMAGE_BUFFER:
    // case GL_IMAGE_1D_ARRAY:
    // case GL_IMAGE_2D_ARRAY:
    case GL_IMAGE_2D_MULTISAMPLE:
      // case GL_IMAGE_2D_MULTISAMPLE_ARRAY:
      return detail::glGetUniformiv(env, program, location, 1);

    default: GLEW_THROW(info.Env(), GL_INVALID_ENUM);
  }
}

// GL_EXPORT GLint glGetUniformLocation (GLuint program, const GLchar* name);
Napi::Value WebGL2RenderingContext::GetUniformLocation(Napi::CallbackInfo const& info) {
  CallbackArgs args   = info;
  GLuint program      = args[0];
  std::string name    = args[1];
  auto const location = GL_EXPORT::glGetUniformLocation(program, name.c_str());
  return location > -1 ? WebGLUniformLocation::New(info.Env(), location) : info.Env().Null();
}

// GL_EXPORT void glUniform1f (GLint location, GLfloat v0);
void WebGL2RenderingContext::Uniform1f(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint loc         = args[0];
  GL_EXPORT::glUniform1f(loc, args[1]);
}

// GL_EXPORT void glUniform1fv (GLint location, GLsizei count, const GLfloat* value);
void WebGL2RenderingContext::Uniform1fv(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint loc         = args[0];
  static constexpr GLint size{1};
  Span<GLfloat> ptr = args[1];
  if ((ptr.size() < size) || (ptr.size() % size) != 0) { GLEW_THROW(info.Env(), GL_INVALID_VALUE); }
  GL_EXPORT::glUniform1fv(loc, ptr.size() / size, ptr.data());
}

// GL_EXPORT void glUniform1i (GLint location, GLint v0);
void WebGL2RenderingContext::Uniform1i(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint loc         = args[0];
  GL_EXPORT::glUniform1i(loc, args[1]);
}

// GL_EXPORT void glUniform1iv (GLint location, GLsizei count, const GLint* value);
void WebGL2RenderingContext::Uniform1iv(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint loc         = args[0];
  static constexpr GLint size{1};
  Span<GLint> ptr = args[1];
  if ((ptr.size() < size) || (ptr.size() % size) != 0) { GLEW_THROW(info.Env(), GL_INVALID_VALUE); }
  GL_EXPORT::glUniform1iv(loc, ptr.size() / size, ptr.data());
}

// GL_EXPORT void glUniform2f (GLint location, GLfloat v0, GLfloat v1);
void WebGL2RenderingContext::Uniform2f(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint loc         = args[0];
  GL_EXPORT::glUniform2f(loc, args[1], args[2]);
}

// GL_EXPORT void glUniform2fv (GLint location, GLsizei count, const GLfloat* value);
void WebGL2RenderingContext::Uniform2fv(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint loc         = args[0];
  static constexpr GLint size{2};
  Span<GLfloat> ptr = args[1];
  if ((ptr.size() < size) || (ptr.size() % size) != 0) { GLEW_THROW(info.Env(), GL_INVALID_VALUE); }
  GL_EXPORT::glUniform2fv(loc, ptr.size() / size, ptr.data());
}

// GL_EXPORT void glUniform2i (GLint location, GLint v0, GLint v1);
void WebGL2RenderingContext::Uniform2i(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint loc         = args[0];
  GL_EXPORT::glUniform2i(loc, args[1], args[2]);
}

// GL_EXPORT void glUniform2iv (GLint location, GLsizei count, const GLint* value);
void WebGL2RenderingContext::Uniform2iv(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint loc         = args[0];
  static constexpr GLint size{2};
  Span<GLint> ptr = args[1];
  if ((ptr.size() < size) || (ptr.size() % size) != 0) { GLEW_THROW(info.Env(), GL_INVALID_VALUE); }
  GL_EXPORT::glUniform2iv(loc, ptr.size() / size, ptr.data());
}

// GL_EXPORT void glUniform3f (GLint location, GLfloat v0, GLfloat v1, GLfloat v2);
void WebGL2RenderingContext::Uniform3f(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint loc         = args[0];
  GL_EXPORT::glUniform3f(loc, args[1], args[2], args[3]);
}

// GL_EXPORT void glUniform3fv (GLint location, GLsizei count, const GLfloat* value);
void WebGL2RenderingContext::Uniform3fv(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint loc         = args[0];
  static constexpr GLint size{3};
  Span<GLfloat> ptr = args[1];
  if ((ptr.size() < size) || (ptr.size() % size) != 0) { GLEW_THROW(info.Env(), GL_INVALID_VALUE); }
  GL_EXPORT::glUniform3fv(loc, ptr.size() / size, ptr.data());
}

// GL_EXPORT void glUniform3i (GLint location, GLint v0, GLint v1, GLint v2);
void WebGL2RenderingContext::Uniform3i(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint loc         = args[0];
  GL_EXPORT::glUniform3i(loc, args[1], args[2], args[3]);
}

// GL_EXPORT void glUniform3iv (GLint location, GLsizei count, const GLint* value);
void WebGL2RenderingContext::Uniform3iv(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint loc         = args[0];
  static constexpr GLint size{3};
  Span<GLint> ptr = args[1];
  if ((ptr.size() < size) || (ptr.size() % size) != 0) { GLEW_THROW(info.Env(), GL_INVALID_VALUE); }
  GL_EXPORT::glUniform3iv(loc, ptr.size() / size, ptr.data());
}

// GL_EXPORT void glUniform4f (GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3);
void WebGL2RenderingContext::Uniform4f(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint loc         = args[0];
  GL_EXPORT::glUniform4f(loc, args[1], args[2], args[3], args[4]);
}

// GL_EXPORT void glUniform4fv (GLint location, GLsizei count, const GLfloat* value);
void WebGL2RenderingContext::Uniform4fv(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint loc         = args[0];
  static constexpr GLint size{4};
  Span<GLfloat> ptr = args[1];
  if ((ptr.size() < size) || (ptr.size() % size) != 0) { GLEW_THROW(info.Env(), GL_INVALID_VALUE); }
  GL_EXPORT::glUniform4fv(loc, ptr.size() / size, ptr.data());
}

// GL_EXPORT void glUniform4i (GLint location, GLint v0, GLint v1, GLint v2, GLint v3);
void WebGL2RenderingContext::Uniform4i(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint loc         = args[0];
  GL_EXPORT::glUniform4i(loc, args[1], args[2], args[3], args[4]);
}

// GL_EXPORT void glUniform4iv (GLint location, GLsizei count, const GLint* value);
void WebGL2RenderingContext::Uniform4iv(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint loc         = args[0];
  static constexpr GLint size{4};
  Span<GLint> ptr = args[1];
  if ((ptr.size() < size) || (ptr.size() % size) != 0) { GLEW_THROW(info.Env(), GL_INVALID_VALUE); }
  GL_EXPORT::glUniform4iv(loc, ptr.size() / size, ptr.data());
}

// GL_EXPORT void glUniformMatrix2fv (GLint location, GLsizei count, GLboolean transpose, const
// GLfloat* value);
void WebGL2RenderingContext::UniformMatrix2fv(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint loc         = args[0];
  bool transpose    = args[1];
  static constexpr GLint size{2 * 2};
  Span<GLfloat> ptr = args[2];
  if ((ptr.size() < size) || (ptr.size() % size) != 0) { GLEW_THROW(info.Env(), GL_INVALID_VALUE); }
  GL_EXPORT::glUniformMatrix2fv(loc, ptr.size() / size, transpose, ptr.data());
}

// GL_EXPORT void glUniformMatrix3fv (GLint location, GLsizei count, GLboolean transpose, const
// GLfloat* value);
void WebGL2RenderingContext::UniformMatrix3fv(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint loc         = args[0];
  bool transpose    = args[1];
  static constexpr GLint size{3 * 3};
  Span<GLfloat> ptr = args[2];
  if ((ptr.size() < size) || (ptr.size() % size) != 0) { GLEW_THROW(info.Env(), GL_INVALID_VALUE); }
  GL_EXPORT::glUniformMatrix3fv(loc, ptr.size() / size, transpose, ptr.data());
}

// GL_EXPORT void glUniformMatrix4fv (GLint location, GLsizei count, GLboolean transpose, const
// GLfloat* value);
void WebGL2RenderingContext::UniformMatrix4fv(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint loc         = args[0];
  bool transpose    = args[1];
  static constexpr GLint size{4 * 4};
  Span<GLfloat> ptr = args[2];
  if ((ptr.size() < size) || (ptr.size() % size) != 0) { GLEW_THROW(info.Env(), GL_INVALID_VALUE); }
  GL_EXPORT::glUniformMatrix4fv(loc, ptr.size() / size, transpose, ptr.data());
}

// GL_EXPORT void glUniformMatrix2x3fv (GLint location, GLsizei count, GLboolean transpose, const
// GLfloat *value);
void WebGL2RenderingContext::UniformMatrix2x3fv(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint loc         = args[0];
  bool transpose    = args[1];
  static constexpr GLint size{2 * 3};
  Span<GLfloat> ptr = args[2];
  if ((ptr.size() < size) || (ptr.size() % size) != 0) { GLEW_THROW(info.Env(), GL_INVALID_VALUE); }
  GL_EXPORT::glUniformMatrix2x3fv(loc, ptr.size() / size, transpose, ptr.data());
}

// GL_EXPORT void glUniformMatrix2x4fv (GLint location, GLsizei count, GLboolean transpose, const
// GLfloat *value);
void WebGL2RenderingContext::UniformMatrix2x4fv(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint loc         = args[0];
  bool transpose    = args[1];
  static constexpr GLint size{2 * 4};
  Span<GLfloat> ptr = args[2];
  if ((ptr.size() < size) || (ptr.size() % size) != 0) { GLEW_THROW(info.Env(), GL_INVALID_VALUE); }
  GL_EXPORT::glUniformMatrix2x4fv(loc, ptr.size() / size, transpose, ptr.data());
}

// GL_EXPORT void glUniformMatrix3x2fv (GLint location, GLsizei count, GLboolean transpose, const
// GLfloat *value);
void WebGL2RenderingContext::UniformMatrix3x2fv(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint loc         = args[0];
  bool transpose    = args[1];
  static constexpr GLint size{3 * 2};
  Span<GLfloat> ptr = args[2];
  if ((ptr.size() < size) || (ptr.size() % size) != 0) { GLEW_THROW(info.Env(), GL_INVALID_VALUE); }
  GL_EXPORT::glUniformMatrix3x2fv(loc, ptr.size() / size, transpose, ptr.data());
}

// GL_EXPORT void glUniformMatrix3x4fv (GLint location, GLsizei count, GLboolean transpose, const
// GLfloat *value);
void WebGL2RenderingContext::UniformMatrix3x4fv(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint loc         = args[0];
  bool transpose    = args[1];
  static constexpr GLint size{3 * 4};
  Span<GLfloat> ptr = args[2];
  if ((ptr.size() < size) || (ptr.size() % size) != 0) { GLEW_THROW(info.Env(), GL_INVALID_VALUE); }
  GL_EXPORT::glUniformMatrix3x4fv(loc, ptr.size() / size, transpose, ptr.data());
}

// GL_EXPORT void glUniformMatrix4x2fv (GLint location, GLsizei count, GLboolean transpose, const
// GLfloat *value);
void WebGL2RenderingContext::UniformMatrix4x2fv(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint loc         = args[0];
  bool transpose    = args[1];
  static constexpr GLint size{4 * 2};
  Span<GLfloat> ptr = args[2];
  if ((ptr.size() < size) || (ptr.size() % size) != 0) { GLEW_THROW(info.Env(), GL_INVALID_VALUE); }
  GL_EXPORT::glUniformMatrix4x2fv(loc, ptr.size() / size, transpose, ptr.data());
}

// GL_EXPORT void glUniformMatrix4x3fv (GLint location, GLsizei count, GLboolean transpose, const
// GLfloat *value);
void WebGL2RenderingContext::UniformMatrix4x3fv(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint loc         = args[0];
  bool transpose    = args[1];
  static constexpr GLint size{4 * 3};
  Span<GLfloat> ptr = args[2];
  if ((ptr.size() < size) || (ptr.size() % size) != 0) { GLEW_THROW(info.Env(), GL_INVALID_VALUE); }
  GL_EXPORT::glUniformMatrix4x3fv(loc, ptr.size() / size, transpose, ptr.data());
}

// GL_EXPORT void glUniform1ui (GLint location, GLuint v0);
void WebGL2RenderingContext::Uniform1ui(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint loc         = args[0];
  GL_EXPORT::glUniform1ui(loc, args[1]);
}

// GL_EXPORT void glUniform1uiv (GLint location, GLsizei count, const GLuint* value);
void WebGL2RenderingContext::Uniform1uiv(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint loc         = args[0];
  static constexpr GLint size{1};
  Span<GLuint> ptr = args[1];
  if ((ptr.size() < size) || (ptr.size() % size) != 0) { GLEW_THROW(info.Env(), GL_INVALID_VALUE); }
  GL_EXPORT::glUniform1uiv(loc, ptr.size() / size, ptr.data());
}

// GL_EXPORT void glUniform2ui (GLint location, GLuint v0, GLuint v1);
void WebGL2RenderingContext::Uniform2ui(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint loc         = args[0];
  GL_EXPORT::glUniform2ui(loc, args[1], args[2]);
}

// GL_EXPORT void glUniform2uiv (GLint location, GLsizei count, const GLuint* value);
void WebGL2RenderingContext::Uniform2uiv(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint loc         = args[0];
  static constexpr GLint size{2};
  Span<GLuint> ptr = args[1];
  if ((ptr.size() < size) || (ptr.size() % size) != 0) { GLEW_THROW(info.Env(), GL_INVALID_VALUE); }
  GL_EXPORT::glUniform2uiv(loc, ptr.size() / size, ptr.data());
}

// GL_EXPORT void glUniform3ui (GLint location, GLuint v0, GLuint v1, GLuint v2);
void WebGL2RenderingContext::Uniform3ui(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint loc         = args[0];
  GL_EXPORT::glUniform3ui(loc, args[1], args[2], args[3]);
}

// GL_EXPORT void glUniform3uiv (GLint location, GLsizei count, const GLuint* value);
void WebGL2RenderingContext::Uniform3uiv(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint loc         = args[0];
  static constexpr GLint size{3};
  Span<GLuint> ptr = args[1];
  if ((ptr.size() < size) || (ptr.size() % size) != 0) { GLEW_THROW(info.Env(), GL_INVALID_VALUE); }
  GL_EXPORT::glUniform3uiv(loc, ptr.size() / size, ptr.data());
}

// GL_EXPORT void glUniform4ui (GLint location, GLuint v0, GLuint v1, GLuint v2, GLuint v3);
void WebGL2RenderingContext::Uniform4ui(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint loc         = args[0];
  GL_EXPORT::glUniform4ui(loc, args[1], args[2], args[3], args[4]);
}

// GL_EXPORT void glUniform4uiv (GLint location, GLsizei count, const GLuint* value);
void WebGL2RenderingContext::Uniform4uiv(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint loc         = args[0];
  static constexpr GLint size{4};
  Span<GLuint> ptr = args[1];
  if ((ptr.size() < size) || (ptr.size() % size) != 0) { GLEW_THROW(info.Env(), GL_INVALID_VALUE); }
  GL_EXPORT::glUniform4uiv(loc, ptr.size() / size, ptr.data());
}

// GL_EXPORT void glGetActiveUniformBlockName (GLuint program, GLuint uniformBlockIndex, GLsizei
// bufSize, GLsizei* length, GLchar* uniformBlockName);
Napi::Value WebGL2RenderingContext::GetActiveUniformBlockName(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLuint program    = args[0];
  GLint max_length{};
  GL_EXPORT::glGetProgramInterfaceiv(program, GL_UNIFORM_BLOCK, GL_MAX_NAME_LENGTH, &max_length);
  if (max_length > 0) {
    GLint length{};
    GLchar* name = reinterpret_cast<GLchar*>(std::malloc(max_length));
    GL_EXPORT::glGetActiveUniformBlockName(program, args[1], max_length, &length, name);
    return CPPToNapi(info.Env())(std::string{name, static_cast<size_t>(length)});
  }
  return info.Env().Null();
}

// GL_EXPORT void glGetActiveUniformBlockiv (GLuint program, GLuint uniformBlockIndex, GLenum pname,
// GLint* params);
Napi::Value WebGL2RenderingContext::GetActiveUniformBlockiv(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint params{};
  GL_EXPORT::glGetActiveUniformBlockiv(args[0], args[1], args[2], &params);
  return CPPToNapi(info.Env())(params);
}

// GL_EXPORT void glGetActiveUniformsiv (GLuint program, GLsizei uniformCount, const GLuint*
// uniformIndices, GLenum pname, GLint* params);
Napi::Value WebGL2RenderingContext::GetActiveUniformsiv(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint params{};
  GL_EXPORT::glGetActiveUniformsiv(args[0], args[1], args[2], args[3], &params);
  return CPPToNapi(info.Env())(params);
}

// GL_EXPORT GLuint glGetUniformBlockIndex (GLuint program, const GLchar* uniformBlockName);
Napi::Value WebGL2RenderingContext::GetUniformBlockIndex(Napi::CallbackInfo const& info) {
  CallbackArgs args            = info;
  GLuint program               = args[0];
  std::string uniformBlockName = args[1];
  return CPPToNapi(info.Env())(GL_EXPORT::glGetUniformBlockIndex(program, uniformBlockName.data()));
}

// GL_EXPORT void glGetUniformIndices (GLuint program, GLsizei uniformCount, const GLchar* const *
// uniformNames, GLuint* uniformIndices);
Napi::Value WebGL2RenderingContext::GetUniformIndices(Napi::CallbackInfo const& info) {
  CallbackArgs args                      = info;
  GLuint program                         = args[0];
  std::vector<std::string> uniform_names = args[1];
  std::vector<GLuint> uniform_indices(uniform_names.size());
  std::vector<const GLchar*> uniform_name_ptrs(uniform_names.size());
  std::transform(uniform_names.begin(),
                 uniform_names.end(),
                 uniform_name_ptrs.begin(),
                 [&](std::string const& str) { return str.data(); });
  GL_EXPORT::glGetUniformIndices(
    program, uniform_names.size(), uniform_name_ptrs.data(), uniform_indices.data());
  return CPPToNapi(info.Env())(uniform_indices);
}

// GL_EXPORT void glUniformBlockBinding (GLuint program, GLuint uniformBlockIndex, GLuint
// uniformBlockBinding);
void WebGL2RenderingContext::UniformBlockBinding(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glUniformBlockBinding(args[0], args[1], args[2]);
}

}  // namespace nv
