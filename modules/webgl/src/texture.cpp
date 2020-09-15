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

// GL_EXPORT void glActiveTexture (GLenum texture);
Napi::Value WebGL2RenderingContext::ActiveTexture(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glActiveTexture(FromJS(info[0]));
  return env.Undefined();
}

// GL_EXPORT void glBindTexture (GLenum target, GLuint texture);
Napi::Value WebGL2RenderingContext::BindTexture(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glBindTexture(FromJS(info[0]), FromJS(info[1]));
  return env.Undefined();
}

// GL_EXPORT void glCompressedTexImage2D (GLenum target, GLint level, GLenum internalformat, GLsizei
// width, GLsizei height, GLint border, GLsizei imageSize, const void *data);
Napi::Value WebGL2RenderingContext::CompressedTexImage2D(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glCompressedTexImage2D(FromJS(info[0]),
                                    FromJS(info[1]),
                                    FromJS(info[2]),
                                    FromJS(info[3]),
                                    FromJS(info[4]),
                                    FromJS(info[5]),
                                    FromJS(info[6]),
                                    FromJS(info[7]));
  return env.Undefined();
}

// GL_EXPORT void glCompressedTexImage3D (GLenum target, GLint level, GLenum internalformat, GLsizei
// width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, const void *data);
Napi::Value WebGL2RenderingContext::CompressedTexImage3D(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glCompressedTexImage3D(FromJS(info[0]),
                                    FromJS(info[1]),
                                    FromJS(info[2]),
                                    FromJS(info[3]),
                                    FromJS(info[4]),
                                    FromJS(info[5]),
                                    FromJS(info[6]),
                                    FromJS(info[7]),
                                    FromJS(info[8]));
  return env.Undefined();
}

// GL_EXPORT void glCompressedTexSubImage2D (GLenum target, GLint level, GLint xoffset, GLint
// yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const void *data);
Napi::Value WebGL2RenderingContext::CompressedTexSubImage2D(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glCompressedTexSubImage2D(FromJS(info[0]),
                                       FromJS(info[1]),
                                       FromJS(info[2]),
                                       FromJS(info[3]),
                                       FromJS(info[4]),
                                       FromJS(info[5]),
                                       FromJS(info[6]),
                                       FromJS(info[7]),
                                       FromJS(info[8]));
  return env.Undefined();
}

// GL_EXPORT void glCopyTexImage2D (GLenum target, GLint level, GLenum internalFormat, GLint x,
// GLint y, GLsizei width, GLsizei height, GLint border);
Napi::Value WebGL2RenderingContext::CopyTexImage2D(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glCopyTexImage2D(FromJS(info[0]),
                              FromJS(info[1]),
                              FromJS(info[2]),
                              FromJS(info[3]),
                              FromJS(info[4]),
                              FromJS(info[5]),
                              FromJS(info[6]),
                              FromJS(info[7]));
  return env.Undefined();
}

// GL_EXPORT void glCopyTexSubImage2D (GLenum target, GLint level, GLint xoffset, GLint yoffset,
// GLint x, GLint y, GLsizei width, GLsizei height);
Napi::Value WebGL2RenderingContext::CopyTexSubImage2D(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glCopyTexSubImage2D(FromJS(info[0]),
                                 FromJS(info[1]),
                                 FromJS(info[2]),
                                 FromJS(info[3]),
                                 FromJS(info[4]),
                                 FromJS(info[5]),
                                 FromJS(info[6]),
                                 FromJS(info[7]));
  return env.Undefined();
}

// GL_EXPORT void glGenTextures (GLsizei n, GLuint* textures);
Napi::Value WebGL2RenderingContext::CreateTexture(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GLuint texture{};
  GL_EXPORT::glGenTextures(1, &texture);
  return WebGLTexture::New(texture);
}

// GL_EXPORT void glGenTextures (GLsizei n, GLuint* textures);
Napi::Value WebGL2RenderingContext::GenTextures(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  std::vector<GLuint> textures(static_cast<size_t>(FromJS(info[0])));
  GL_EXPORT::glGenTextures(textures.size(), textures.data());
  return ToNapi(env)(textures);
}

// GL_EXPORT void glDeleteTextures (GLsizei n, const GLuint *textures);
Napi::Value WebGL2RenderingContext::DeleteTexture(Napi::CallbackInfo const& info) {
  auto env       = info.Env();
  GLuint texture = FromJS(info[0]);
  GL_EXPORT::glDeleteTextures(1, &texture);
  return env.Undefined();
}

// GL_EXPORT void glDeleteTextures (GLsizei n, const GLuint *textures);
Napi::Value WebGL2RenderingContext::DeleteTextures(Napi::CallbackInfo const& info) {
  auto env                     = info.Env();
  std::vector<GLuint> textures = FromJS(info[0]);
  GL_EXPORT::glDeleteTextures(textures.size(), textures.data());
  return env.Undefined();
}

// GL_EXPORT void glGenerateMipmap (GLenum target);
Napi::Value WebGL2RenderingContext::GenerateMipmap(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glGenerateMipmap(FromJS(info[0]));
  return env.Undefined();
}

// GL_EXPORT void glGetTexParameterfv (GLenum target, GLenum pname, GLfloat *params);
// GL_EXPORT void glGetTexParameteriv (GLenum target, GLenum pname, GLint *params);
// GL_EXPORT void glGetTexParameterIiv (GLenum target, GLenum pname, GLint* params);
// GL_EXPORT void glGetTexParameterIuiv (GLenum target, GLenum pname, GLuint* params);
Napi::Value WebGL2RenderingContext::GetTexParameter(Napi::CallbackInfo const& info) {
  auto env     = info.Env();
  GLint target = FromJS(info[0]);
  GLint pname  = FromJS(info[1]);
  switch (pname) {
    case GL_TEXTURE_IMMUTABLE_FORMAT: {
      GLint params{};
      GL_EXPORT::glGetTexParameteriv(target, pname, &params);
      return ToNapi(env)(static_cast<bool>(params));
    }
    case GL_TEXTURE_MAG_FILTER:
    case GL_TEXTURE_MIN_FILTER:
    case GL_TEXTURE_WRAP_S:
    case GL_TEXTURE_WRAP_T:
    case GL_TEXTURE_COMPARE_FUNC:
    case GL_TEXTURE_COMPARE_MODE:
    case GL_TEXTURE_WRAP_R:
    case GL_TEXTURE_BASE_LEVEL:
    case GL_TEXTURE_MAX_LEVEL: {
      GLint params{};
      GL_EXPORT::glGetTexParameteriv(target, pname, &params);
      return ToNapi(env)(params);
    }
    case GL_TEXTURE_IMMUTABLE_LEVELS: {
      GLuint params{};
      GL_EXPORT::glGetTexParameterIuiv(target, pname, &params);
      return ToNapi(env)(params);
    }
    case GL_TEXTURE_MAX_LOD:
    case GL_TEXTURE_MIN_LOD:
    case GL_TEXTURE_MAX_ANISOTROPY_EXT: {
      GLfloat params{};
      GL_EXPORT::glGetTexParameterfv(target, pname, &params);
      return ToNapi(env)(params);
    }
    default: GLEW_THROW(env, GL_INVALID_ENUM);
  }
}

// GL_EXPORT GLboolean glIsTexture (GLuint texture);
Napi::Value WebGL2RenderingContext::IsTexture(Napi::CallbackInfo const& info) {
  return ToNapi(info.Env())(GL_EXPORT::glIsTexture(FromJS(info[0])));
}

// GL_EXPORT void glTexImage2D (GLenum target, GLint level, GLint internalformat, GLsizei width,
// GLsizei height, GLint border, GLenum format, GLenum type, const void *pixels);
Napi::Value WebGL2RenderingContext::TexImage2D(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  if (info.Length() < 9 || info[8].IsEmpty() || info[8].IsNull()) {
    GL_EXPORT::glTexImage2D(FromJS(info[0]),
                            FromJS(info[1]),
                            FromJS(info[2]),
                            FromJS(info[3]),
                            FromJS(info[4]),
                            FromJS(info[5]),
                            FromJS(info[6]),
                            FromJS(info[7]),
                            nullptr);
  } else if (info[8].IsNumber()) {
    GLint offset = FromJS(info[8]);
    GL_EXPORT::glTexImage2D(FromJS(info[0]),
                            FromJS(info[1]),
                            FromJS(info[2]),
                            FromJS(info[3]),
                            FromJS(info[4]),
                            FromJS(info[5]),
                            FromJS(info[6]),
                            FromJS(info[7]),
                            reinterpret_cast<void*>(offset));
  } else {
    void* pixels = FromJS(info[8]);
    GL_EXPORT::glTexImage2D(FromJS(info[0]),
                            FromJS(info[1]),
                            FromJS(info[2]),
                            FromJS(info[3]),
                            FromJS(info[4]),
                            FromJS(info[5]),
                            FromJS(info[6]),
                            FromJS(info[7]),
                            pixels);
  }
  return env.Undefined();
}

// GL_EXPORT void glTexParameterf (GLenum target, GLenum pname, GLfloat param);
Napi::Value WebGL2RenderingContext::TexParameterf(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glTexParameterf(FromJS(info[0]), FromJS(info[1]), FromJS(info[2]));
  return env.Undefined();
}

// GL_EXPORT void glTexParameteri (GLenum target, GLenum pname, GLint param);
Napi::Value WebGL2RenderingContext::TexParameteri(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glTexParameteri(FromJS(info[0]), FromJS(info[1]), FromJS(info[2]));
  return env.Undefined();
}

// GL_EXPORT void glTexSubImage2D (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei
// width, GLsizei height, GLenum format, GLenum type, const void *pixels);
Napi::Value WebGL2RenderingContext::TexSubImage2D(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  if (info.Length() < 9 || info[8].IsEmpty() || info[8].IsNull()) {
    GL_EXPORT::glTexSubImage2D(FromJS(info[0]),
                               FromJS(info[1]),
                               FromJS(info[2]),
                               FromJS(info[3]),
                               FromJS(info[4]),
                               FromJS(info[5]),
                               FromJS(info[6]),
                               FromJS(info[7]),
                               nullptr);
  } else if (info[8].IsNumber()) {
    GLint offset = FromJS(info[8]);
    GL_EXPORT::glTexSubImage2D(FromJS(info[0]),
                               FromJS(info[1]),
                               FromJS(info[2]),
                               FromJS(info[3]),
                               FromJS(info[4]),
                               FromJS(info[5]),
                               FromJS(info[6]),
                               FromJS(info[7]),
                               reinterpret_cast<void*>(offset));
  } else {
    void* pixels = FromJS(info[8]);
    GL_EXPORT::glTexSubImage2D(FromJS(info[0]),
                               FromJS(info[1]),
                               FromJS(info[2]),
                               FromJS(info[3]),
                               FromJS(info[4]),
                               FromJS(info[5]),
                               FromJS(info[6]),
                               FromJS(info[7]),
                               pixels);
  }
  return env.Undefined();
}

// GL_EXPORT void glCompressedTexSubImage3D (GLenum target, GLint level, GLint xoffset, GLint
// yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei
// imageSize, const void *data);
Napi::Value WebGL2RenderingContext::CompressedTexSubImage3D(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glCompressedTexSubImage3D(FromJS(info[0]),
                                       FromJS(info[1]),
                                       FromJS(info[2]),
                                       FromJS(info[3]),
                                       FromJS(info[4]),
                                       FromJS(info[5]),
                                       FromJS(info[6]),
                                       FromJS(info[7]),
                                       FromJS(info[8]),
                                       FromJS(info[9]),
                                       FromJS(info[10]));
  return env.Undefined();
}

// GL_EXPORT void glCopyTexSubImage3D (GLenum target, GLint level, GLint xoffset, GLint yoffset,
// GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height);
Napi::Value WebGL2RenderingContext::CopyTexSubImage3D(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glCopyTexSubImage3D(FromJS(info[0]),
                                 FromJS(info[1]),
                                 FromJS(info[2]),
                                 FromJS(info[3]),
                                 FromJS(info[4]),
                                 FromJS(info[5]),
                                 FromJS(info[6]),
                                 FromJS(info[7]),
                                 FromJS(info[8]));
  return env.Undefined();
}

// GL_EXPORT void glTexImage3D (GLenum target, GLint level, GLint internalFormat, GLsizei width,
// GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const void *pixels);
Napi::Value WebGL2RenderingContext::TexImage3D(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glTexImage3D(FromJS(info[0]),
                          FromJS(info[1]),
                          FromJS(info[2]),
                          FromJS(info[3]),
                          FromJS(info[4]),
                          FromJS(info[5]),
                          FromJS(info[6]),
                          FromJS(info[7]),
                          FromJS(info[8]),
                          FromJS(info[9]));
  return env.Undefined();
}

// GL_EXPORT void glTexStorage2D (GLenum target, GLsizei levels, GLenum internalformat, GLsizei
// width, GLsizei height);
Napi::Value WebGL2RenderingContext::TexStorage2D(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glTexStorage2D(
    FromJS(info[0]), FromJS(info[1]), FromJS(info[2]), FromJS(info[3]), FromJS(info[4]));
  return env.Undefined();
}

// GL_EXPORT void glTexStorage3D (GLenum target, GLsizei levels, GLenum internalformat, GLsizei
// width, GLsizei height, GLsizei depth);
Napi::Value WebGL2RenderingContext::TexStorage3D(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glTexStorage3D(FromJS(info[0]),
                            FromJS(info[1]),
                            FromJS(info[2]),
                            FromJS(info[3]),
                            FromJS(info[4]),
                            FromJS(info[5]));
  return env.Undefined();
}

// GL_EXPORT void glTexSubImage3D (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint
// zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void
// *pixels);
Napi::Value WebGL2RenderingContext::TexSubImage3D(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glTexSubImage3D(FromJS(info[0]),
                             FromJS(info[1]),
                             FromJS(info[2]),
                             FromJS(info[3]),
                             FromJS(info[4]),
                             FromJS(info[5]),
                             FromJS(info[6]),
                             FromJS(info[7]),
                             FromJS(info[8]),
                             FromJS(info[9]),
                             FromJS(info[10]));
  return env.Undefined();
}

}  // namespace node_webgl
