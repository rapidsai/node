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

// GL_EXPORT void glActiveTexture (GLenum texture);
void WebGL2RenderingContext::ActiveTexture(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glActiveTexture(args[0]);
}

// GL_EXPORT void glBindTexture (GLenum target, GLuint texture);
void WebGL2RenderingContext::BindTexture(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glBindTexture(args[0], args[1]);
}

// GL_EXPORT void glCompressedTexImage2D (GLenum target, GLint level, GLenum internalformat, GLsizei
// width, GLsizei height, GLint border, GLsizei imageSize, const void *data);
void WebGL2RenderingContext::CompressedTexImage2D(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glCompressedTexImage2D(
    args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]);
}

// GL_EXPORT void glCompressedTexImage3D (GLenum target, GLint level, GLenum internalformat, GLsizei
// width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, const void *data);
void WebGL2RenderingContext::CompressedTexImage3D(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glCompressedTexImage3D(
    args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8]);
}

// GL_EXPORT void glCompressedTexSubImage2D (GLenum target, GLint level, GLint xoffset, GLint
// yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const void *data);
void WebGL2RenderingContext::CompressedTexSubImage2D(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glCompressedTexSubImage2D(
    args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8]);
}

// GL_EXPORT void glCopyTexImage2D (GLenum target, GLint level, GLenum internalFormat, GLint x,
// GLint y, GLsizei width, GLsizei height, GLint border);
void WebGL2RenderingContext::CopyTexImage2D(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glCopyTexImage2D(
    args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]);
}

// GL_EXPORT void glCopyTexSubImage2D (GLenum target, GLint level, GLint xoffset, GLint yoffset,
// GLint x, GLint y, GLsizei width, GLsizei height);
void WebGL2RenderingContext::CopyTexSubImage2D(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glCopyTexSubImage2D(
    args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]);
}

// GL_EXPORT void glGenTextures (GLsizei n, GLuint* textures);
Napi::Value WebGL2RenderingContext::CreateTexture(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLuint texture{};
  GL_EXPORT::glGenTextures(1, &texture);
  return WebGLTexture::New(info.Env(), texture);
}

// GL_EXPORT void glGenTextures (GLsizei n, GLuint* textures);
Napi::Value WebGL2RenderingContext::GenTextures(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  std::vector<GLuint> textures(static_cast<size_t>(args[0]));
  GL_EXPORT::glGenTextures(textures.size(), textures.data());
  return CPPToNapi(info)(textures);
}

// GL_EXPORT void glDeleteTextures (GLsizei n, const GLuint *textures);
void WebGL2RenderingContext::DeleteTexture(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLuint texture    = args[0];
  GL_EXPORT::glDeleteTextures(1, &texture);
}

// GL_EXPORT void glDeleteTextures (GLsizei n, const GLuint *textures);
void WebGL2RenderingContext::DeleteTextures(Napi::CallbackInfo const& info) {
  CallbackArgs args            = info;
  std::vector<GLuint> textures = args[0];
  GL_EXPORT::glDeleteTextures(textures.size(), textures.data());
}

// GL_EXPORT void glGenerateMipmap (GLenum target);
void WebGL2RenderingContext::GenerateMipmap(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glGenerateMipmap(args[0]);
}

// GL_EXPORT void glGetTexParameterfv (GLenum target, GLenum pname, GLfloat *params);
// GL_EXPORT void glGetTexParameteriv (GLenum target, GLenum pname, GLint *params);
// GL_EXPORT void glGetTexParameterIiv (GLenum target, GLenum pname, GLint* params);
// GL_EXPORT void glGetTexParameterIuiv (GLenum target, GLenum pname, GLuint* params);
Napi::Value WebGL2RenderingContext::GetTexParameter(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint target      = args[0];
  GLint pname       = args[1];
  switch (pname) {
    case GL_TEXTURE_IMMUTABLE_FORMAT: {
      GLint params{};
      GL_EXPORT::glGetTexParameteriv(target, pname, &params);
      return CPPToNapi(info)(static_cast<bool>(params));
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
      return CPPToNapi(info)(params);
    }
    case GL_TEXTURE_IMMUTABLE_LEVELS: {
      GLuint params{};
      GL_EXPORT::glGetTexParameterIuiv(target, pname, &params);
      return CPPToNapi(info)(params);
    }
    case GL_TEXTURE_MAX_LOD:
    case GL_TEXTURE_MIN_LOD:
    case GL_TEXTURE_MAX_ANISOTROPY_EXT: {
      GLfloat params{};
      GL_EXPORT::glGetTexParameterfv(target, pname, &params);
      return CPPToNapi(info)(params);
    }
    default: GLEW_THROW(info.Env(), GL_INVALID_ENUM);
  }
}

// GL_EXPORT GLboolean glIsTexture (GLuint texture);
Napi::Value WebGL2RenderingContext::IsTexture(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  auto is_texture   = GL_EXPORT::glIsTexture(args[0]);
  return CPPToNapi(info.Env())(is_texture);
}

// GL_EXPORT void glTexImage2D (GLenum target, GLint level, GLint internalformat, GLsizei width,
// GLsizei height, GLint border, GLenum format, GLenum type, const void *pixels);
void WebGL2RenderingContext::TexImage2D(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  if (info.Length() < 9 || info[8].IsEmpty() || info[8].IsNull()) {
    GL_EXPORT::glTexImage2D(
      args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], nullptr);
  } else if (info[8].IsNumber()) {
    GLint offset = args[8];
    GL_EXPORT::glTexImage2D(args[0],
                            args[1],
                            args[2],
                            args[3],
                            args[4],
                            args[5],
                            args[6],
                            args[7],
                            reinterpret_cast<void*>(offset));
  } else {
    void* pixels = args[8];
    GL_EXPORT::glTexImage2D(
      args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], pixels);
  }
}

// GL_EXPORT void glTexParameterf (GLenum target, GLenum pname, GLfloat param);
void WebGL2RenderingContext::TexParameterf(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glTexParameterf(args[0], args[1], args[2]);
}

// GL_EXPORT void glTexParameteri (GLenum target, GLenum pname, GLint param);
void WebGL2RenderingContext::TexParameteri(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glTexParameteri(args[0], args[1], args[2]);
}

// GL_EXPORT void glTexSubImage2D (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei
// width, GLsizei height, GLenum format, GLenum type, const void *pixels);
void WebGL2RenderingContext::TexSubImage2D(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  if (info.Length() < 9 || info[8].IsEmpty() || info[8].IsNull()) {
    GL_EXPORT::glTexSubImage2D(
      args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], nullptr);
  } else if (info[8].IsNumber()) {
    GLint offset = args[8];
    GL_EXPORT::glTexSubImage2D(args[0],
                               args[1],
                               args[2],
                               args[3],
                               args[4],
                               args[5],
                               args[6],
                               args[7],
                               reinterpret_cast<void*>(offset));
  } else {
    void* pixels = args[8];
    GL_EXPORT::glTexSubImage2D(
      args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], pixels);
  }
}

// GL_EXPORT void glCompressedTexSubImage3D (GLenum target, GLint level, GLint xoffset, GLint
// yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei
// imageSize, const void *data);
void WebGL2RenderingContext::CompressedTexSubImage3D(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glCompressedTexSubImage3D(args[0],
                                       args[1],
                                       args[2],
                                       args[3],
                                       args[4],
                                       args[5],
                                       args[6],
                                       args[7],
                                       args[8],
                                       args[9],
                                       args[10]);
}

// GL_EXPORT void glCopyTexSubImage3D (GLenum target, GLint level, GLint xoffset, GLint yoffset,
// GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height);
void WebGL2RenderingContext::CopyTexSubImage3D(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glCopyTexSubImage3D(
    args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8]);
}

// GL_EXPORT void glTexImage3D (GLenum target, GLint level, GLint internalFormat, GLsizei width,
// GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const void *pixels);
void WebGL2RenderingContext::TexImage3D(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glTexImage3D(
    args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9]);
}

// GL_EXPORT void glTexStorage2D (GLenum target, GLsizei levels, GLenum internalformat, GLsizei
// width, GLsizei height);
void WebGL2RenderingContext::TexStorage2D(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glTexStorage2D(args[0], args[1], args[2], args[3], args[4]);
}

// GL_EXPORT void glTexStorage3D (GLenum target, GLsizei levels, GLenum internalformat, GLsizei
// width, GLsizei height, GLsizei depth);
void WebGL2RenderingContext::TexStorage3D(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glTexStorage3D(args[0], args[1], args[2], args[3], args[4], args[5]);
}

// GL_EXPORT void glTexSubImage3D (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint
// zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void
// *pixels);
void WebGL2RenderingContext::TexSubImage3D(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glTexSubImage3D(args[0],
                             args[1],
                             args[2],
                             args[3],
                             args[4],
                             args[5],
                             args[6],
                             args[7],
                             args[8],
                             args[9],
                             args[10]);
}

}  // namespace nv
