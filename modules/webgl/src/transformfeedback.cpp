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

#include "macros.hpp"
#include "webgl.hpp"

#include <nv_node/utilities/args.hpp>
#include <nv_node/utilities/cpp_to_napi.hpp>

namespace nv {

// GL_EXPORT void glBeginTransformFeedback (GLenum primitiveMode);
Napi::Value WebGL2RenderingContext::BeginTransformFeedback(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glBeginTransformFeedback(args[0]);
  return info.Env().Undefined();
}

// GL_EXPORT void glBindTransformFeedback (GLenum target, GLuint id);
Napi::Value WebGL2RenderingContext::BindTransformFeedback(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glBindTransformFeedback(args[0], args[1]);
  return info.Env().Undefined();
}

// GL_EXPORT void glCreateTransformFeedbacks (GLsizei n, GLuint* ids);
Napi::Value WebGL2RenderingContext::CreateTransformFeedback(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLuint transform_feedback{};
  GL_EXPORT::glCreateTransformFeedbacks(1, &transform_feedback);
  return WebGLTransformFeedback::New(transform_feedback);
}

// GL_EXPORT void glCreateTransformFeedbacks (GLsizei n, GLuint* ids);
Napi::Value WebGL2RenderingContext::CreateTransformFeedbacks(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  std::vector<GLuint> transform_feedbacks(static_cast<size_t>(args[0]));
  GL_EXPORT::glCreateTransformFeedbacks(transform_feedbacks.size(), transform_feedbacks.data());
  return CPPToNapi(info)(transform_feedbacks);
}

// GL_EXPORT void glDeleteTransformFeedbacks (GLsizei n, const GLuint* ids);
Napi::Value WebGL2RenderingContext::DeleteTransformFeedback(Napi::CallbackInfo const& info) {
  CallbackArgs args         = info;
  GLuint transform_feedback = args[0];
  GL_EXPORT::glDeleteTransformFeedbacks(1, &transform_feedback);
  return info.Env().Undefined();
}

// GL_EXPORT void glDeleteTransformFeedbacks (GLsizei n, const GLuint* ids);
Napi::Value WebGL2RenderingContext::DeleteTransformFeedbacks(Napi::CallbackInfo const& info) {
  CallbackArgs args                       = info;
  std::vector<GLuint> transform_feedbacks = args[0];
  GL_EXPORT::glDeleteTransformFeedbacks(transform_feedbacks.size(), transform_feedbacks.data());
  return info.Env().Undefined();
}

// GL_EXPORT void glEndTransformFeedback (void);
Napi::Value WebGL2RenderingContext::EndTransformFeedback(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glEndTransformFeedback();
  return info.Env().Undefined();
}

// GL_EXPORT void glGetTransformFeedbackVarying (GLuint program, GLuint index, GLsizei bufSize,
// GLsizei * length, GLsizei * size, GLenum * type, GLchar * name);
Napi::Value WebGL2RenderingContext::GetTransformFeedbackVarying(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLuint program    = args[0];
  GLuint location   = args[1];
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
  return info.Env().Null();
}

// GL_EXPORT GLboolean glIsTransformFeedback (GLuint id);
Napi::Value WebGL2RenderingContext::IsTransformFeedback(Napi::CallbackInfo const& info) {
  CallbackArgs args          = info;
  auto is_transform_feedback = GL_EXPORT::glIsTransformFeedback(args[0]);
  return CPPToNapi(info)(is_transform_feedback);
}

// GL_EXPORT void glPauseTransformFeedback (void);
Napi::Value WebGL2RenderingContext::PauseTransformFeedback(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glPauseTransformFeedback();
  return info.Env().Undefined();
}

// GL_EXPORT void glResumeTransformFeedback (void);
Napi::Value WebGL2RenderingContext::ResumeTransformFeedback(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glResumeTransformFeedback();
  return info.Env().Undefined();
}

// GL_EXPORT void glTransformFeedbackVaryings (GLuint program, GLsizei count, const GLchar *const*
// varyings, GLenum bufferMode);
Napi::Value WebGL2RenderingContext::TransformFeedbackVaryings(Napi::CallbackInfo const& info) {
  CallbackArgs args                 = info;
  std::vector<std::string> varyings = args[1];
  std::vector<const GLchar*> varying_ptrs(varyings.size());
  std::transform(
    varyings.begin(), varyings.end(), varying_ptrs.begin(), [&](const std::string& str) {
      return str.data();
    });
  GL_EXPORT::glTransformFeedbackVaryings(args[0], varyings.size(), varying_ptrs.data(), args[2]);
  return info.Env().Undefined();
}

}  // namespace nv
