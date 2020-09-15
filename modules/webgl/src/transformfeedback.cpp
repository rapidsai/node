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

// GL_EXPORT void glBeginTransformFeedback (GLenum primitiveMode);
Napi::Value WebGL2RenderingContext::BeginTransformFeedback(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glBeginTransformFeedback(FromJS(info[0]));
  return env.Undefined();
}

// GL_EXPORT void glBindTransformFeedback (GLenum target, GLuint id);
Napi::Value WebGL2RenderingContext::BindTransformFeedback(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glBindTransformFeedback(FromJS(info[0]), FromJS(info[1]));
  return env.Undefined();
}

// GL_EXPORT void glCreateTransformFeedbacks (GLsizei n, GLuint* ids);
Napi::Value WebGL2RenderingContext::CreateTransformFeedback(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GLuint transform_feedback{};
  GL_EXPORT::glCreateTransformFeedbacks(1, &transform_feedback);
  return WebGLTransformFeedback::New(transform_feedback);
}

// GL_EXPORT void glCreateTransformFeedbacks (GLsizei n, GLuint* ids);
Napi::Value WebGL2RenderingContext::CreateTransformFeedbacks(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  std::vector<GLuint> transform_feedbacks(static_cast<size_t>(FromJS(info[0])));
  GL_EXPORT::glCreateTransformFeedbacks(transform_feedbacks.size(), transform_feedbacks.data());
  return ToNapi(env)(transform_feedbacks);
}

// GL_EXPORT void glDeleteTransformFeedbacks (GLsizei n, const GLuint* ids);
Napi::Value WebGL2RenderingContext::DeleteTransformFeedback(Napi::CallbackInfo const& info) {
  auto env                  = info.Env();
  GLuint transform_feedback = FromJS(info[0]);
  GL_EXPORT::glDeleteTransformFeedbacks(1, &transform_feedback);
  return env.Undefined();
}

// GL_EXPORT void glDeleteTransformFeedbacks (GLsizei n, const GLuint* ids);
Napi::Value WebGL2RenderingContext::DeleteTransformFeedbacks(Napi::CallbackInfo const& info) {
  auto env                                = info.Env();
  std::vector<GLuint> transform_feedbacks = FromJS(info[0]);
  GL_EXPORT::glDeleteTransformFeedbacks(transform_feedbacks.size(), transform_feedbacks.data());
  return env.Undefined();
}

// GL_EXPORT void glEndTransformFeedback (void);
Napi::Value WebGL2RenderingContext::EndTransformFeedback(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glEndTransformFeedback();
  return env.Undefined();
}

// GL_EXPORT void glGetTransformFeedbackVarying (GLuint program, GLuint index, GLsizei bufSize,
// GLsizei * length, GLsizei * size, GLenum * type, GLchar * name);
Napi::Value WebGL2RenderingContext::GetTransformFeedbackVarying(Napi::CallbackInfo const& info) {
  auto env        = info.Env();
  GLuint program  = FromJS(info[0]);
  GLuint location = FromJS(info[1]);
  GLint max_length{};
  GL_EXPORT::glGetProgramiv(program, GL_TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH, &max_length);
  if (max_length > 0) {
    GLuint type{};
    GLint size{}, len{};
    GLchar* name = reinterpret_cast<GLchar*>(std::malloc(max_length));
    GL_EXPORT::glGetTransformFeedbackVarying(
      program, location, max_length, &len, &size, &type, name);
    return WebGLActiveInfo::New(size, type, std::string{name, static_cast<size_t>(len)});
  }
  return env.Null();
}

// GL_EXPORT GLboolean glIsTransformFeedback (GLuint id);
Napi::Value WebGL2RenderingContext::IsTransformFeedback(Napi::CallbackInfo const& info) {
  return ToNapi(info.Env())(GL_EXPORT::glIsTransformFeedback(FromJS(info[0])));
}

// GL_EXPORT void glPauseTransformFeedback (void);
Napi::Value WebGL2RenderingContext::PauseTransformFeedback(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glPauseTransformFeedback();
  return env.Undefined();
}

// GL_EXPORT void glResumeTransformFeedback (void);
Napi::Value WebGL2RenderingContext::ResumeTransformFeedback(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glResumeTransformFeedback();
  return env.Undefined();
}

// GL_EXPORT void glTransformFeedbackVaryings (GLuint program, GLsizei count, const GLchar *const*
// varyings, GLenum bufferMode);
Napi::Value WebGL2RenderingContext::TransformFeedbackVaryings(Napi::CallbackInfo const& info) {
  auto env                          = info.Env();
  std::vector<std::string> varyings = FromJS(info[1]);
  std::vector<const GLchar*> varying_ptrs(varyings.size());
  std::transform(
    varyings.begin(), varyings.end(), varying_ptrs.begin(), [&](const std::string& str) {
      return str.data();
    });
  GL_EXPORT::glTransformFeedbackVaryings(
    FromJS(info[0]), varyings.size(), varying_ptrs.data(), FromJS(info[2]));
  return env.Undefined();
}

}  // namespace node_webgl
