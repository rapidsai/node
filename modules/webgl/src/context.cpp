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

#include <iterator>
#include <sstream>

namespace nv {

Napi::FunctionReference WebGL2RenderingContext::constructor;

WebGL2RenderingContext::WebGL2RenderingContext(Napi::CallbackInfo const& info)
  : Napi::ObjectWrap<WebGL2RenderingContext>(info) {
  glewExperimental = GL_TRUE;
  GL_EXPECT_OK(Env(), GLEWAPIENTRY::glewInit());
  if (info[0].IsNull() || info[0].IsEmpty() || !info[0].IsObject()) {
    context_attributes_ = Napi::Persistent(Napi::Object::New(Env()));
  } else {
    context_attributes_ = Napi::Persistent(info[0].As<Napi::Object>());
  }

  // TODO: Is this necessary?
  //
  // auto attrs = context_attributes_.Value();
  // if (attrs.Has("antialias") && attrs.Get("antialias").ToBoolean().Value() == true) {
  //   GL_EXPORT::glEnable(GL_LINE_SMOOTH);
  //   GL_EXPORT::glEnable(GL_POLYGON_SMOOTH);
  // }
  //
  // GL_EXPORT::glEnable(GL_PROGRAM_POINT_SIZE);
  // GL_EXPORT::glEnable(GL_POINT_SPRITE);
};

Napi::Object WebGL2RenderingContext::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor = DefineClass(
    env,
    "WebGL2RenderingContext",
    {
      InstanceAccessor("_clearMask",
                       &WebGL2RenderingContext::GetClearMask_,
                       &WebGL2RenderingContext::SetClearMask_,
                       napi_default),

#define INST_METHOD(key, func) \
  InstanceMethod(              \
    key,                       \
    func,                      \
    static_cast<napi_property_attributes>(napi_writable | napi_enumerable | napi_configurable))

      // INST_METHOD("isSupported", &WebGL2RenderingContext::IsSupported),
      // INST_METHOD("getExtension", &WebGL2RenderingContext::GetExtension),
      // INST_METHOD("getString", &WebGL2RenderingContext::GetString),
      // INST_METHOD("getErrorString", &WebGL2RenderingContext::GetErrorString),
      INST_METHOD("clear", &WebGL2RenderingContext::Clear),
      INST_METHOD("clearColor", &WebGL2RenderingContext::ClearColor),
      INST_METHOD("clearDepth", &WebGL2RenderingContext::ClearDepth),
      INST_METHOD("colorMask", &WebGL2RenderingContext::ColorMask),
      INST_METHOD("cullFace", &WebGL2RenderingContext::CullFace),
      INST_METHOD("depthFunc", &WebGL2RenderingContext::DepthFunc),
      INST_METHOD("depthMask", &WebGL2RenderingContext::DepthMask),
      INST_METHOD("depthRange", &WebGL2RenderingContext::DepthRange),
      INST_METHOD("disable", &WebGL2RenderingContext::Disable),
      INST_METHOD("drawArrays", &WebGL2RenderingContext::DrawArrays),
      INST_METHOD("drawElements", &WebGL2RenderingContext::DrawElements),
      INST_METHOD("enable", &WebGL2RenderingContext::Enable),
      INST_METHOD("finish", &WebGL2RenderingContext::Finish),
      INST_METHOD("flush", &WebGL2RenderingContext::Flush),
      INST_METHOD("frontFace", &WebGL2RenderingContext::FrontFace),
      INST_METHOD("getError", &WebGL2RenderingContext::GetError),
      INST_METHOD("hint", &WebGL2RenderingContext::Hint),
      INST_METHOD("isEnabled", &WebGL2RenderingContext::IsEnabled),
      INST_METHOD("lineWidth", &WebGL2RenderingContext::LineWidth),
      INST_METHOD("pixelStorei", &WebGL2RenderingContext::PixelStorei),
      INST_METHOD("polygonOffset", &WebGL2RenderingContext::PolygonOffset),
      INST_METHOD("readPixels", &WebGL2RenderingContext::ReadPixels),
      INST_METHOD("scissor", &WebGL2RenderingContext::Scissor),
      INST_METHOD("viewport", &WebGL2RenderingContext::Viewport),
      INST_METHOD("drawRangeElements", &WebGL2RenderingContext::DrawRangeElements),
      INST_METHOD("sampleCoverage", &WebGL2RenderingContext::SampleCoverage),
      INST_METHOD("getContextAttributes", &WebGL2RenderingContext::GetContextAttributes),
      INST_METHOD("getFragDataLocation", &WebGL2RenderingContext::GetFragDataLocation),
      INST_METHOD("getParameter", &WebGL2RenderingContext::GetParameter),
      INST_METHOD("getSupportedExtensions", &WebGL2RenderingContext::GetSupportedExtensions),
      INST_METHOD("bindAttribLocation", &WebGL2RenderingContext::BindAttribLocation),
      INST_METHOD("disableVertexAttribArray", &WebGL2RenderingContext::DisableVertexAttribArray),
      INST_METHOD("enableVertexAttribArray", &WebGL2RenderingContext::EnableVertexAttribArray),
      INST_METHOD("getActiveAttrib", &WebGL2RenderingContext::GetActiveAttrib),
      INST_METHOD("getAttribLocation", &WebGL2RenderingContext::GetAttribLocation),
      INST_METHOD("getVertexAttrib", &WebGL2RenderingContext::GetVertexAttrib),
      INST_METHOD("getVertexAttribOffset", &WebGL2RenderingContext::GetVertexAttribPointerv),
      INST_METHOD("vertexAttrib1f", &WebGL2RenderingContext::VertexAttrib1f),
      INST_METHOD("vertexAttrib1fv", &WebGL2RenderingContext::VertexAttrib1fv),
      INST_METHOD("vertexAttrib2f", &WebGL2RenderingContext::VertexAttrib2f),
      INST_METHOD("vertexAttrib2fv", &WebGL2RenderingContext::VertexAttrib2fv),
      INST_METHOD("vertexAttrib3f", &WebGL2RenderingContext::VertexAttrib3f),
      INST_METHOD("vertexAttrib3fv", &WebGL2RenderingContext::VertexAttrib3fv),
      INST_METHOD("vertexAttrib4f", &WebGL2RenderingContext::VertexAttrib4f),
      INST_METHOD("vertexAttrib4fv", &WebGL2RenderingContext::VertexAttrib4fv),
      INST_METHOD("vertexAttribPointer", &WebGL2RenderingContext::VertexAttribPointer),
      INST_METHOD("vertexAttribI4i", &WebGL2RenderingContext::VertexAttribI4i),
      INST_METHOD("vertexAttribI4iv", &WebGL2RenderingContext::VertexAttribI4iv),
      INST_METHOD("vertexAttribI4ui", &WebGL2RenderingContext::VertexAttribI4ui),
      INST_METHOD("vertexAttribI4uiv", &WebGL2RenderingContext::VertexAttribI4uiv),
      INST_METHOD("vertexAttribIPointer", &WebGL2RenderingContext::VertexAttribIPointer),
      INST_METHOD("blendColor", &WebGL2RenderingContext::BlendColor),
      INST_METHOD("blendEquation", &WebGL2RenderingContext::BlendEquation),
      INST_METHOD("blendEquationSeparate", &WebGL2RenderingContext::BlendEquationSeparate),
      INST_METHOD("blendFunc", &WebGL2RenderingContext::BlendFunc),
      INST_METHOD("blendFuncSeparate", &WebGL2RenderingContext::BlendFuncSeparate),
      INST_METHOD("bindBuffer", &WebGL2RenderingContext::BindBuffer),
      INST_METHOD("bufferData", &WebGL2RenderingContext::BufferData),
      INST_METHOD("bufferSubData", &WebGL2RenderingContext::BufferSubData),
      INST_METHOD("createBuffer", &WebGL2RenderingContext::CreateBuffer),
      INST_METHOD("createBuffers", &WebGL2RenderingContext::CreateBuffers),
      INST_METHOD("deleteBuffer", &WebGL2RenderingContext::DeleteBuffer),
      INST_METHOD("deleteBuffers", &WebGL2RenderingContext::DeleteBuffers),
      INST_METHOD("getBufferParameter", &WebGL2RenderingContext::GetBufferParameter),
      INST_METHOD("isBuffer", &WebGL2RenderingContext::IsBuffer),
      INST_METHOD("bindBufferBase", &WebGL2RenderingContext::BindBufferBase),
      INST_METHOD("bindBufferRange", &WebGL2RenderingContext::BindBufferRange),
      INST_METHOD("clearBufferfv", &WebGL2RenderingContext::ClearBufferfv),
      INST_METHOD("clearBufferiv", &WebGL2RenderingContext::ClearBufferiv),
      INST_METHOD("clearBufferuiv", &WebGL2RenderingContext::ClearBufferuiv),
      INST_METHOD("clearBufferfi", &WebGL2RenderingContext::ClearBufferfi),
      INST_METHOD("copyBufferSubData", &WebGL2RenderingContext::CopyBufferSubData),
      INST_METHOD("drawBuffers", &WebGL2RenderingContext::DrawBuffers),
      INST_METHOD("getBufferSubData", &WebGL2RenderingContext::GetBufferSubData),
      INST_METHOD("readBuffer", &WebGL2RenderingContext::ReadBuffer),
      INST_METHOD("bindFramebuffer", &WebGL2RenderingContext::BindFramebuffer),
      INST_METHOD("checkFramebufferStatus", &WebGL2RenderingContext::CheckFramebufferStatus),
      INST_METHOD("createFramebuffer", &WebGL2RenderingContext::CreateFramebuffer),
      INST_METHOD("createFramebuffers", &WebGL2RenderingContext::CreateFramebuffers),
      INST_METHOD("deleteFramebuffer", &WebGL2RenderingContext::DeleteFramebuffer),
      INST_METHOD("deleteFramebuffers", &WebGL2RenderingContext::DeleteFramebuffers),
      INST_METHOD("framebufferRenderbuffer", &WebGL2RenderingContext::FramebufferRenderbuffer),
      INST_METHOD("framebufferTexture2D", &WebGL2RenderingContext::FramebufferTexture2D),
      INST_METHOD("getFramebufferAttachmentParameter",
                  &WebGL2RenderingContext::GetFramebufferAttachmentParameter),
      INST_METHOD("isFramebuffer", &WebGL2RenderingContext::IsFramebuffer),
      INST_METHOD("blitFramebuffer", &WebGL2RenderingContext::BlitFramebuffer),
      INST_METHOD("framebufferTextureLayer", &WebGL2RenderingContext::FramebufferTextureLayer),
      INST_METHOD("invalidateFramebuffer", &WebGL2RenderingContext::InvalidateFramebuffer),
      INST_METHOD("invalidateSubFramebuffer", &WebGL2RenderingContext::InvalidateSubFramebuffer),
      INST_METHOD("drawArraysInstanced", &WebGL2RenderingContext::DrawArraysInstanced),
      INST_METHOD("drawElementsInstanced", &WebGL2RenderingContext::DrawElementsInstanced),
      INST_METHOD("vertexAttribDivisor", &WebGL2RenderingContext::VertexAttribDivisor),
      INST_METHOD("createProgram", &WebGL2RenderingContext::CreateProgram),
      INST_METHOD("deleteProgram", &WebGL2RenderingContext::DeleteProgram),
      INST_METHOD("getProgramInfoLog", &WebGL2RenderingContext::GetProgramInfoLog),
      INST_METHOD("getProgramParameter", &WebGL2RenderingContext::GetProgramParameter),
      INST_METHOD("isProgram", &WebGL2RenderingContext::IsProgram),
      INST_METHOD("linkProgram", &WebGL2RenderingContext::LinkProgram),
      INST_METHOD("useProgram", &WebGL2RenderingContext::UseProgram),
      INST_METHOD("validateProgram", &WebGL2RenderingContext::ValidateProgram),
      INST_METHOD("beginQuery", &WebGL2RenderingContext::BeginQuery),
      INST_METHOD("createQuery", &WebGL2RenderingContext::CreateQuery),
      INST_METHOD("createQueries", &WebGL2RenderingContext::CreateQueries),
      INST_METHOD("deleteQuery", &WebGL2RenderingContext::DeleteQuery),
      INST_METHOD("deleteQueries", &WebGL2RenderingContext::DeleteQueries),
      INST_METHOD("endQuery", &WebGL2RenderingContext::EndQuery),
      INST_METHOD("getQuery", &WebGL2RenderingContext::GetQuery),
      INST_METHOD("getQueryParameter", &WebGL2RenderingContext::GetQueryParameter),
      INST_METHOD("isQuery", &WebGL2RenderingContext::IsQuery),
      INST_METHOD("bindSampler", &WebGL2RenderingContext::BindSampler),
      INST_METHOD("createSampler", &WebGL2RenderingContext::CreateSampler),
      INST_METHOD("createSamplers", &WebGL2RenderingContext::CreateSamplers),
      INST_METHOD("deleteSampler", &WebGL2RenderingContext::DeleteSampler),
      INST_METHOD("deleteSamplers", &WebGL2RenderingContext::DeleteSamplers),
      INST_METHOD("getSamplerParameter", &WebGL2RenderingContext::GetSamplerParameter),
      INST_METHOD("isSampler", &WebGL2RenderingContext::IsSampler),
      INST_METHOD("samplerParameterf", &WebGL2RenderingContext::SamplerParameterf),
      INST_METHOD("samplerParameteri", &WebGL2RenderingContext::SamplerParameteri),
      INST_METHOD("attachShader", &WebGL2RenderingContext::AttachShader),
      INST_METHOD("compileShader", &WebGL2RenderingContext::CompileShader),
      INST_METHOD("createShader", &WebGL2RenderingContext::CreateShader),
      INST_METHOD("deleteShader", &WebGL2RenderingContext::DeleteShader),
      INST_METHOD("detachShader", &WebGL2RenderingContext::DetachShader),
      INST_METHOD("getAttachedShaders", &WebGL2RenderingContext::GetAttachedShaders),
      INST_METHOD("getShaderInfoLog", &WebGL2RenderingContext::GetShaderInfoLog),
      INST_METHOD("getShaderParameter", &WebGL2RenderingContext::GetShaderParameter),
      INST_METHOD("getShaderPrecisionFormat", &WebGL2RenderingContext::GetShaderPrecisionFormat),
      INST_METHOD("getShaderSource", &WebGL2RenderingContext::GetShaderSource),
      INST_METHOD("isShader", &WebGL2RenderingContext::IsShader),
      INST_METHOD("shaderSource", &WebGL2RenderingContext::ShaderSource),
      INST_METHOD("clearStencil", &WebGL2RenderingContext::ClearStencil),
      INST_METHOD("stencilFunc", &WebGL2RenderingContext::StencilFunc),
      INST_METHOD("stencilFuncSeparate", &WebGL2RenderingContext::StencilFuncSeparate),
      INST_METHOD("stencilMask", &WebGL2RenderingContext::StencilMask),
      INST_METHOD("stencilMaskSeparate", &WebGL2RenderingContext::StencilMaskSeparate),
      INST_METHOD("stencilOp", &WebGL2RenderingContext::StencilOp),
      INST_METHOD("stencilOpSeparate", &WebGL2RenderingContext::StencilOpSeparate),
      INST_METHOD("clientWaitSync", &WebGL2RenderingContext::ClientWaitSync),
      INST_METHOD("deleteSync", &WebGL2RenderingContext::DeleteSync),
      INST_METHOD("fenceSync", &WebGL2RenderingContext::FenceSync),
      INST_METHOD("getSyncParameter", &WebGL2RenderingContext::GetSyncParameter),
      INST_METHOD("isSync", &WebGL2RenderingContext::IsSync),
      INST_METHOD("waitSync", &WebGL2RenderingContext::WaitSync),
      INST_METHOD("bindRenderbuffer", &WebGL2RenderingContext::BindRenderbuffer),
      INST_METHOD("createRenderbuffer", &WebGL2RenderingContext::CreateRenderbuffer),
      INST_METHOD("createRenderbuffers", &WebGL2RenderingContext::CreateRenderbuffers),
      INST_METHOD("deleteRenderbuffer", &WebGL2RenderingContext::DeleteRenderbuffer),
      INST_METHOD("deleteRenderbuffers", &WebGL2RenderingContext::DeleteRenderbuffers),
      INST_METHOD("getRenderbufferParameteriv",
                  &WebGL2RenderingContext::GetRenderbufferParameteriv),
      INST_METHOD("isRenderbuffer", &WebGL2RenderingContext::IsRenderbuffer),
      INST_METHOD("renderbufferStorage", &WebGL2RenderingContext::RenderbufferStorage),
      INST_METHOD("renderbufferStorageMultisample",
                  &WebGL2RenderingContext::RenderbufferStorageMultisample),
      INST_METHOD("activeTexture", &WebGL2RenderingContext::ActiveTexture),
      INST_METHOD("bindTexture", &WebGL2RenderingContext::BindTexture),
      INST_METHOD("compressedTexImage2D", &WebGL2RenderingContext::CompressedTexImage2D),
      INST_METHOD("compressedTexImage3D", &WebGL2RenderingContext::CompressedTexImage3D),
      INST_METHOD("compressedTexSubImage2D", &WebGL2RenderingContext::CompressedTexSubImage2D),
      INST_METHOD("copyTexImage2D", &WebGL2RenderingContext::CopyTexImage2D),
      INST_METHOD("copyTexSubImage2D", &WebGL2RenderingContext::CopyTexSubImage2D),
      INST_METHOD("createTexture", &WebGL2RenderingContext::CreateTexture),
      INST_METHOD("genTextures", &WebGL2RenderingContext::GenTextures),
      INST_METHOD("deleteTexture", &WebGL2RenderingContext::DeleteTexture),
      INST_METHOD("deleteTextures", &WebGL2RenderingContext::DeleteTextures),
      INST_METHOD("generateMipmap", &WebGL2RenderingContext::GenerateMipmap),
      INST_METHOD("getTexParameter", &WebGL2RenderingContext::GetTexParameter),
      INST_METHOD("isTexture", &WebGL2RenderingContext::IsTexture),
      INST_METHOD("texImage2D", &WebGL2RenderingContext::TexImage2D),
      INST_METHOD("texParameterf", &WebGL2RenderingContext::TexParameterf),
      INST_METHOD("texParameteri", &WebGL2RenderingContext::TexParameteri),
      INST_METHOD("texSubImage2D", &WebGL2RenderingContext::TexSubImage2D),
      INST_METHOD("compressedTexSubImage3D", &WebGL2RenderingContext::CompressedTexSubImage3D),
      INST_METHOD("copyTexSubImage3D", &WebGL2RenderingContext::CopyTexSubImage3D),
      INST_METHOD("texImage3D", &WebGL2RenderingContext::TexImage3D),
      INST_METHOD("texStorage2D", &WebGL2RenderingContext::TexStorage2D),
      INST_METHOD("texStorage3D", &WebGL2RenderingContext::TexStorage3D),
      INST_METHOD("texSubImage3D", &WebGL2RenderingContext::TexSubImage3D),
      INST_METHOD("beginTransformFeedback", &WebGL2RenderingContext::BeginTransformFeedback),
      INST_METHOD("bindTransformFeedback", &WebGL2RenderingContext::BindTransformFeedback),
      INST_METHOD("createTransformFeedback", &WebGL2RenderingContext::CreateTransformFeedback),
      INST_METHOD("createTransformFeedbacks", &WebGL2RenderingContext::CreateTransformFeedbacks),
      INST_METHOD("deleteTransformFeedback", &WebGL2RenderingContext::DeleteTransformFeedback),
      INST_METHOD("deleteTransformFeedbacks", &WebGL2RenderingContext::DeleteTransformFeedbacks),
      INST_METHOD("endTransformFeedback", &WebGL2RenderingContext::EndTransformFeedback),
      INST_METHOD("getTransformFeedbackVarying",
                  &WebGL2RenderingContext::GetTransformFeedbackVarying),
      INST_METHOD("isTransformFeedback", &WebGL2RenderingContext::IsTransformFeedback),
      INST_METHOD("pauseTransformFeedback", &WebGL2RenderingContext::PauseTransformFeedback),
      INST_METHOD("resumeTransformFeedback", &WebGL2RenderingContext::ResumeTransformFeedback),
      INST_METHOD("transformFeedbackVaryings", &WebGL2RenderingContext::TransformFeedbackVaryings),
      INST_METHOD("getActiveUniform", &WebGL2RenderingContext::GetActiveUniform),
      INST_METHOD("getUniform", &WebGL2RenderingContext::GetUniform),
      INST_METHOD("getUniformLocation", &WebGL2RenderingContext::GetUniformLocation),
      INST_METHOD("uniform1f", &WebGL2RenderingContext::Uniform1f),
      INST_METHOD("uniform1fv", &WebGL2RenderingContext::Uniform1fv),
      INST_METHOD("uniform1i", &WebGL2RenderingContext::Uniform1i),
      INST_METHOD("uniform1iv", &WebGL2RenderingContext::Uniform1iv),
      INST_METHOD("uniform2f", &WebGL2RenderingContext::Uniform2f),
      INST_METHOD("uniform2fv", &WebGL2RenderingContext::Uniform2fv),
      INST_METHOD("uniform2i", &WebGL2RenderingContext::Uniform2i),
      INST_METHOD("uniform2iv", &WebGL2RenderingContext::Uniform2iv),
      INST_METHOD("uniform3f", &WebGL2RenderingContext::Uniform3f),
      INST_METHOD("uniform3fv", &WebGL2RenderingContext::Uniform3fv),
      INST_METHOD("uniform3i", &WebGL2RenderingContext::Uniform3i),
      INST_METHOD("uniform3iv", &WebGL2RenderingContext::Uniform3iv),
      INST_METHOD("uniform4f", &WebGL2RenderingContext::Uniform4f),
      INST_METHOD("uniform4fv", &WebGL2RenderingContext::Uniform4fv),
      INST_METHOD("uniform4i", &WebGL2RenderingContext::Uniform4i),
      INST_METHOD("uniform4iv", &WebGL2RenderingContext::Uniform4iv),
      INST_METHOD("uniformMatrix2fv", &WebGL2RenderingContext::UniformMatrix2fv),
      INST_METHOD("uniformMatrix3fv", &WebGL2RenderingContext::UniformMatrix3fv),
      INST_METHOD("uniformMatrix4fv", &WebGL2RenderingContext::UniformMatrix4fv),
      INST_METHOD("uniformMatrix2x3fv", &WebGL2RenderingContext::UniformMatrix2x3fv),
      INST_METHOD("uniformMatrix2x4fv", &WebGL2RenderingContext::UniformMatrix2x4fv),
      INST_METHOD("uniformMatrix3x2fv", &WebGL2RenderingContext::UniformMatrix3x2fv),
      INST_METHOD("uniformMatrix3x4fv", &WebGL2RenderingContext::UniformMatrix3x4fv),
      INST_METHOD("uniformMatrix4x2fv", &WebGL2RenderingContext::UniformMatrix4x2fv),
      INST_METHOD("uniformMatrix4x3fv", &WebGL2RenderingContext::UniformMatrix4x3fv),
      INST_METHOD("uniform1ui", &WebGL2RenderingContext::Uniform1ui),
      INST_METHOD("uniform1uiv", &WebGL2RenderingContext::Uniform1uiv),
      INST_METHOD("uniform2ui", &WebGL2RenderingContext::Uniform2ui),
      INST_METHOD("uniform2uiv", &WebGL2RenderingContext::Uniform2uiv),
      INST_METHOD("uniform3ui", &WebGL2RenderingContext::Uniform3ui),
      INST_METHOD("uniform3uiv", &WebGL2RenderingContext::Uniform3uiv),
      INST_METHOD("uniform4ui", &WebGL2RenderingContext::Uniform4ui),
      INST_METHOD("uniform4uiv", &WebGL2RenderingContext::Uniform4uiv),
      INST_METHOD("getActiveUniformBlockName", &WebGL2RenderingContext::GetActiveUniformBlockName),
      INST_METHOD("getActiveUniformBlockParameter",
                  &WebGL2RenderingContext::GetActiveUniformBlockiv),
      INST_METHOD("getActiveUniforms", &WebGL2RenderingContext::GetActiveUniformsiv),
      INST_METHOD("getUniformBlockIndex", &WebGL2RenderingContext::GetUniformBlockIndex),
      INST_METHOD("getUniformIndices", &WebGL2RenderingContext::GetUniformIndices),
      INST_METHOD("uniformBlockBinding", &WebGL2RenderingContext::UniformBlockBinding),
      INST_METHOD("createVertexArray", &WebGL2RenderingContext::CreateVertexArray),
      INST_METHOD("createVertexArrays", &WebGL2RenderingContext::CreateVertexArrays),
      INST_METHOD("bindVertexArray", &WebGL2RenderingContext::BindVertexArray),
      INST_METHOD("deleteVertexArray", &WebGL2RenderingContext::DeleteVertexArray),
      INST_METHOD("deleteVertexArrays", &WebGL2RenderingContext::DeleteVertexArrays),
      INST_METHOD("isVertexArray", &WebGL2RenderingContext::IsVertexArray),
#undef INST_METHOD

#define INST_ENUM(key, val) \
  InstanceValue(key, Napi::Number::New(env, static_cast<int64_t>(val)), napi_enumerable)

      INST_ENUM("READ_BUFFER", GL_READ_BUFFER),
      INST_ENUM("UNPACK_ROW_LENGTH", GL_UNPACK_ROW_LENGTH),
      INST_ENUM("UNPACK_SKIP_ROWS", GL_UNPACK_SKIP_ROWS),
      INST_ENUM("UNPACK_SKIP_PIXELS", GL_UNPACK_SKIP_PIXELS),
      INST_ENUM("PACK_ROW_LENGTH", GL_PACK_ROW_LENGTH),
      INST_ENUM("PACK_SKIP_ROWS", GL_PACK_SKIP_ROWS),
      INST_ENUM("PACK_SKIP_PIXELS", GL_PACK_SKIP_PIXELS),
      INST_ENUM("COLOR", GL_COLOR),
      INST_ENUM("DEPTH", GL_DEPTH),
      INST_ENUM("STENCIL", GL_STENCIL),
      INST_ENUM("RED", GL_RED),
      INST_ENUM("RGB8", GL_RGB8),
      INST_ENUM("RGBA8", GL_RGBA8),
      INST_ENUM("RGB10_A2", GL_RGB10_A2),
      INST_ENUM("TEXTURE_BINDING_3D", GL_TEXTURE_BINDING_3D),
      INST_ENUM("UNPACK_SKIP_IMAGES", GL_UNPACK_SKIP_IMAGES),
      INST_ENUM("UNPACK_IMAGE_HEIGHT", GL_UNPACK_IMAGE_HEIGHT),
      INST_ENUM("TEXTURE_3D", GL_TEXTURE_3D),
      INST_ENUM("TEXTURE_WRAP_R", GL_TEXTURE_WRAP_R),
      INST_ENUM("MAX_3D_TEXTURE_SIZE", GL_MAX_3D_TEXTURE_SIZE),
      INST_ENUM("UNSIGNED_INT_2_10_10_10_REV", GL_UNSIGNED_INT_2_10_10_10_REV),
      INST_ENUM("MAX_ELEMENTS_VERTICES", GL_MAX_ELEMENTS_VERTICES),
      INST_ENUM("MAX_ELEMENTS_INDICES", GL_MAX_ELEMENTS_INDICES),
      INST_ENUM("TEXTURE_MIN_LOD", GL_TEXTURE_MIN_LOD),
      INST_ENUM("TEXTURE_MAX_LOD", GL_TEXTURE_MAX_LOD),
      INST_ENUM("TEXTURE_BASE_LEVEL", GL_TEXTURE_BASE_LEVEL),
      INST_ENUM("TEXTURE_MAX_LEVEL", GL_TEXTURE_MAX_LEVEL),
      INST_ENUM("MIN", GL_MIN),
      INST_ENUM("MAX", GL_MAX),
      INST_ENUM("DEPTH_COMPONENT24", GL_DEPTH_COMPONENT24),
      INST_ENUM("MAX_TEXTURE_LOD_BIAS", GL_MAX_TEXTURE_LOD_BIAS),
      INST_ENUM("TEXTURE_COMPARE_MODE", GL_TEXTURE_COMPARE_MODE),
      INST_ENUM("TEXTURE_COMPARE_FUNC", GL_TEXTURE_COMPARE_FUNC),
      INST_ENUM("CURRENT_QUERY", GL_CURRENT_QUERY),
      INST_ENUM("QUERY_RESULT", GL_QUERY_RESULT),
      INST_ENUM("QUERY_RESULT_AVAILABLE", GL_QUERY_RESULT_AVAILABLE),
      INST_ENUM("STREAM_READ", GL_STREAM_READ),
      INST_ENUM("STREAM_COPY", GL_STREAM_COPY),
      INST_ENUM("STATIC_READ", GL_STATIC_READ),
      INST_ENUM("STATIC_COPY", GL_STATIC_COPY),
      INST_ENUM("DYNAMIC_READ", GL_DYNAMIC_READ),
      INST_ENUM("DYNAMIC_COPY", GL_DYNAMIC_COPY),
      INST_ENUM("MAX_DRAW_BUFFERS", GL_MAX_DRAW_BUFFERS),
      INST_ENUM("DRAW_BUFFER0", GL_DRAW_BUFFER0),
      INST_ENUM("DRAW_BUFFER1", GL_DRAW_BUFFER1),
      INST_ENUM("DRAW_BUFFER2", GL_DRAW_BUFFER2),
      INST_ENUM("DRAW_BUFFER3", GL_DRAW_BUFFER3),
      INST_ENUM("DRAW_BUFFER4", GL_DRAW_BUFFER4),
      INST_ENUM("DRAW_BUFFER5", GL_DRAW_BUFFER5),
      INST_ENUM("DRAW_BUFFER6", GL_DRAW_BUFFER6),
      INST_ENUM("DRAW_BUFFER7", GL_DRAW_BUFFER7),
      INST_ENUM("DRAW_BUFFER8", GL_DRAW_BUFFER8),
      INST_ENUM("DRAW_BUFFER9", GL_DRAW_BUFFER9),
      INST_ENUM("DRAW_BUFFER10", GL_DRAW_BUFFER10),
      INST_ENUM("DRAW_BUFFER11", GL_DRAW_BUFFER11),
      INST_ENUM("DRAW_BUFFER12", GL_DRAW_BUFFER12),
      INST_ENUM("DRAW_BUFFER13", GL_DRAW_BUFFER13),
      INST_ENUM("DRAW_BUFFER14", GL_DRAW_BUFFER14),
      INST_ENUM("DRAW_BUFFER15", GL_DRAW_BUFFER15),
      INST_ENUM("MAX_FRAGMENT_UNIFORM_COMPONENTS", GL_MAX_FRAGMENT_UNIFORM_COMPONENTS),
      INST_ENUM("MAX_VERTEX_UNIFORM_COMPONENTS", GL_MAX_VERTEX_UNIFORM_COMPONENTS),
      INST_ENUM("SAMPLER_3D", GL_SAMPLER_3D),
      INST_ENUM("SAMPLER_2D_SHADOW", GL_SAMPLER_2D_SHADOW),
      INST_ENUM("FRAGMENT_SHADER_DERIVATIVE_HINT", GL_FRAGMENT_SHADER_DERIVATIVE_HINT),
      INST_ENUM("PIXEL_PACK_BUFFER", GL_PIXEL_PACK_BUFFER),
      INST_ENUM("PIXEL_UNPACK_BUFFER", GL_PIXEL_UNPACK_BUFFER),
      INST_ENUM("PIXEL_PACK_BUFFER_BINDING", GL_PIXEL_PACK_BUFFER_BINDING),
      INST_ENUM("PIXEL_UNPACK_BUFFER_BINDING", GL_PIXEL_UNPACK_BUFFER_BINDING),
      INST_ENUM("SRGB", GL_SRGB),
      INST_ENUM("SRGB8", GL_SRGB8),
      INST_ENUM("SRGB8_ALPHA8", GL_SRGB8_ALPHA8),
      INST_ENUM("COMPARE_REF_TO_TEXTURE", GL_COMPARE_REF_TO_TEXTURE),
      INST_ENUM("RGBA32F", GL_RGBA32F),
      INST_ENUM("RGB32F", GL_RGB32F),
      INST_ENUM("RGBA16F", GL_RGBA16F),
      INST_ENUM("RGB16F", GL_RGB16F),
      INST_ENUM("VERTEX_ATTRIB_ARRAY_INTEGER", GL_VERTEX_ATTRIB_ARRAY_INTEGER),
      INST_ENUM("MAX_ARRAY_TEXTURE_LAYERS", GL_MAX_ARRAY_TEXTURE_LAYERS),
      INST_ENUM("MIN_PROGRAM_TEXEL_OFFSET", GL_MIN_PROGRAM_TEXEL_OFFSET),
      INST_ENUM("MAX_PROGRAM_TEXEL_OFFSET", GL_MAX_PROGRAM_TEXEL_OFFSET),
      INST_ENUM("MAX_VARYING_COMPONENTS", GL_MAX_VARYING_COMPONENTS),
      INST_ENUM("TEXTURE_2D_ARRAY", GL_TEXTURE_2D_ARRAY),
      INST_ENUM("TEXTURE_BINDING_2D_ARRAY", GL_TEXTURE_BINDING_2D_ARRAY),
      INST_ENUM("R11F_G11F_B10F", GL_R11F_G11F_B10F),
      INST_ENUM("UNSIGNED_INT_10F_11F_11F_REV", GL_UNSIGNED_INT_10F_11F_11F_REV),
      INST_ENUM("RGB9_E5", GL_RGB9_E5),
      INST_ENUM("UNSIGNED_INT_5_9_9_9_REV", GL_UNSIGNED_INT_5_9_9_9_REV),
      INST_ENUM("TRANSFORM_FEEDBACK_BUFFER_MODE", GL_TRANSFORM_FEEDBACK_BUFFER_MODE),
      INST_ENUM("MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS",
                GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS),
      INST_ENUM("TRANSFORM_FEEDBACK_VARYINGS", GL_TRANSFORM_FEEDBACK_VARYINGS),
      INST_ENUM("TRANSFORM_FEEDBACK_BUFFER_START", GL_TRANSFORM_FEEDBACK_BUFFER_START),
      INST_ENUM("TRANSFORM_FEEDBACK_BUFFER_SIZE", GL_TRANSFORM_FEEDBACK_BUFFER_SIZE),
      INST_ENUM("TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN", GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN),
      INST_ENUM("RASTERIZER_DISCARD", GL_RASTERIZER_DISCARD),
      INST_ENUM("MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS",
                GL_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS),
      INST_ENUM("MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS",
                GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS),
      INST_ENUM("INTERLEAVED_ATTRIBS", GL_INTERLEAVED_ATTRIBS),
      INST_ENUM("SEPARATE_ATTRIBS", GL_SEPARATE_ATTRIBS),
      INST_ENUM("TRANSFORM_FEEDBACK_BUFFER", GL_TRANSFORM_FEEDBACK_BUFFER),
      INST_ENUM("TRANSFORM_FEEDBACK_BUFFER_BINDING", GL_TRANSFORM_FEEDBACK_BUFFER_BINDING),
      INST_ENUM("RGBA32UI", GL_RGBA32UI),
      INST_ENUM("RGB32UI", GL_RGB32UI),
      INST_ENUM("RGBA16UI", GL_RGBA16UI),
      INST_ENUM("RGB16UI", GL_RGB16UI),
      INST_ENUM("RGBA8UI", GL_RGBA8UI),
      INST_ENUM("RGB8UI", GL_RGB8UI),
      INST_ENUM("RGBA32I", GL_RGBA32I),
      INST_ENUM("RGB32I", GL_RGB32I),
      INST_ENUM("RGBA16I", GL_RGBA16I),
      INST_ENUM("RGB16I", GL_RGB16I),
      INST_ENUM("RGBA8I", GL_RGBA8I),
      INST_ENUM("RGB8I", GL_RGB8I),
      INST_ENUM("RED_INTEGER", GL_RED_INTEGER),
      INST_ENUM("RGB_INTEGER", GL_RGB_INTEGER),
      INST_ENUM("RGBA_INTEGER", GL_RGBA_INTEGER),
      INST_ENUM("SAMPLER_2D_ARRAY", GL_SAMPLER_2D_ARRAY),
      INST_ENUM("SAMPLER_2D_ARRAY_SHADOW", GL_SAMPLER_2D_ARRAY_SHADOW),
      INST_ENUM("SAMPLER_CUBE_SHADOW", GL_SAMPLER_CUBE_SHADOW),
      INST_ENUM("UNSIGNED_INT_VEC2", GL_UNSIGNED_INT_VEC2),
      INST_ENUM("UNSIGNED_INT_VEC3", GL_UNSIGNED_INT_VEC3),
      INST_ENUM("UNSIGNED_INT_VEC4", GL_UNSIGNED_INT_VEC4),
      INST_ENUM("INT_SAMPLER_2D", GL_INT_SAMPLER_2D),
      INST_ENUM("INT_SAMPLER_3D", GL_INT_SAMPLER_3D),
      INST_ENUM("INT_SAMPLER_CUBE", GL_INT_SAMPLER_CUBE),
      INST_ENUM("INT_SAMPLER_2D_ARRAY", GL_INT_SAMPLER_2D_ARRAY),
      INST_ENUM("UNSIGNED_INT_SAMPLER_2D", GL_UNSIGNED_INT_SAMPLER_2D),
      INST_ENUM("UNSIGNED_INT_SAMPLER_3D", GL_UNSIGNED_INT_SAMPLER_3D),
      INST_ENUM("UNSIGNED_INT_SAMPLER_CUBE", GL_UNSIGNED_INT_SAMPLER_CUBE),
      INST_ENUM("UNSIGNED_INT_SAMPLER_2D_ARRAY", GL_UNSIGNED_INT_SAMPLER_2D_ARRAY),
      INST_ENUM("DEPTH_COMPONENT32F", GL_DEPTH_COMPONENT32F),
      INST_ENUM("DEPTH32F_STENCIL8", GL_DEPTH32F_STENCIL8),
      INST_ENUM("FLOAT_32_UNSIGNED_INT_24_8_REV", GL_FLOAT_32_UNSIGNED_INT_24_8_REV),
      INST_ENUM("FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING", GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING),
      INST_ENUM("FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE", GL_FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE),
      INST_ENUM("FRAMEBUFFER_ATTACHMENT_RED_SIZE", GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE),
      INST_ENUM("FRAMEBUFFER_ATTACHMENT_GREEN_SIZE", GL_FRAMEBUFFER_ATTACHMENT_GREEN_SIZE),
      INST_ENUM("FRAMEBUFFER_ATTACHMENT_BLUE_SIZE", GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE),
      INST_ENUM("FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE", GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE),
      INST_ENUM("FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE", GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE),
      INST_ENUM("FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE", GL_FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE),
      INST_ENUM("FRAMEBUFFER_DEFAULT", GL_FRAMEBUFFER_DEFAULT),
      INST_ENUM("UNSIGNED_INT_24_8", GL_UNSIGNED_INT_24_8),
      INST_ENUM("DEPTH24_STENCIL8", GL_DEPTH24_STENCIL8),
      INST_ENUM("UNSIGNED_NORMALIZED", GL_UNSIGNED_NORMALIZED),
      INST_ENUM("DRAW_FRAMEBUFFER_BINDING", GL_DRAW_FRAMEBUFFER_BINDING),
      INST_ENUM("READ_FRAMEBUFFER", GL_READ_FRAMEBUFFER),
      INST_ENUM("DRAW_FRAMEBUFFER", GL_DRAW_FRAMEBUFFER),
      INST_ENUM("READ_FRAMEBUFFER_BINDING", GL_READ_FRAMEBUFFER_BINDING),
      INST_ENUM("RENDERBUFFER_SAMPLES", GL_RENDERBUFFER_SAMPLES),
      INST_ENUM("FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER", GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER),
      INST_ENUM("MAX_COLOR_ATTACHMENTS", GL_MAX_COLOR_ATTACHMENTS),
      INST_ENUM("COLOR_ATTACHMENT1", GL_COLOR_ATTACHMENT1),
      INST_ENUM("COLOR_ATTACHMENT2", GL_COLOR_ATTACHMENT2),
      INST_ENUM("COLOR_ATTACHMENT3", GL_COLOR_ATTACHMENT3),
      INST_ENUM("COLOR_ATTACHMENT4", GL_COLOR_ATTACHMENT4),
      INST_ENUM("COLOR_ATTACHMENT5", GL_COLOR_ATTACHMENT5),
      INST_ENUM("COLOR_ATTACHMENT6", GL_COLOR_ATTACHMENT6),
      INST_ENUM("COLOR_ATTACHMENT7", GL_COLOR_ATTACHMENT7),
      INST_ENUM("COLOR_ATTACHMENT8", GL_COLOR_ATTACHMENT8),
      INST_ENUM("COLOR_ATTACHMENT9", GL_COLOR_ATTACHMENT9),
      INST_ENUM("COLOR_ATTACHMENT10", GL_COLOR_ATTACHMENT10),
      INST_ENUM("COLOR_ATTACHMENT11", GL_COLOR_ATTACHMENT11),
      INST_ENUM("COLOR_ATTACHMENT12", GL_COLOR_ATTACHMENT12),
      INST_ENUM("COLOR_ATTACHMENT13", GL_COLOR_ATTACHMENT13),
      INST_ENUM("COLOR_ATTACHMENT14", GL_COLOR_ATTACHMENT14),
      INST_ENUM("COLOR_ATTACHMENT15", GL_COLOR_ATTACHMENT15),
      INST_ENUM("FRAMEBUFFER_INCOMPLETE_MULTISAMPLE", GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE),
      INST_ENUM("MAX_SAMPLES", GL_MAX_SAMPLES),
      INST_ENUM("HALF_FLOAT", GL_HALF_FLOAT),
      INST_ENUM("RG", GL_RG),
      INST_ENUM("RG_INTEGER", GL_RG_INTEGER),
      INST_ENUM("R8", GL_R8),
      INST_ENUM("RG8", GL_RG8),
      INST_ENUM("R16F", GL_R16F),
      INST_ENUM("R32F", GL_R32F),
      INST_ENUM("RG16F", GL_RG16F),
      INST_ENUM("RG32F", GL_RG32F),
      INST_ENUM("R8I", GL_R8I),
      INST_ENUM("R8UI", GL_R8UI),
      INST_ENUM("R16I", GL_R16I),
      INST_ENUM("R16UI", GL_R16UI),
      INST_ENUM("R32I", GL_R32I),
      INST_ENUM("R32UI", GL_R32UI),
      INST_ENUM("RG8I", GL_RG8I),
      INST_ENUM("RG8UI", GL_RG8UI),
      INST_ENUM("RG16I", GL_RG16I),
      INST_ENUM("RG16UI", GL_RG16UI),
      INST_ENUM("RG32I", GL_RG32I),
      INST_ENUM("RG32UI", GL_RG32UI),
      INST_ENUM("VERTEX_ARRAY_BINDING", GL_VERTEX_ARRAY_BINDING),
      INST_ENUM("R8_SNORM", GL_R8_SNORM),
      INST_ENUM("RG8_SNORM", GL_RG8_SNORM),
      INST_ENUM("RGB8_SNORM", GL_RGB8_SNORM),
      INST_ENUM("RGBA8_SNORM", GL_RGBA8_SNORM),
      INST_ENUM("SIGNED_NORMALIZED", GL_SIGNED_NORMALIZED),
      INST_ENUM("COPY_READ_BUFFER", GL_COPY_READ_BUFFER),
      INST_ENUM("COPY_WRITE_BUFFER", GL_COPY_WRITE_BUFFER),
      INST_ENUM("COPY_READ_BUFFER_BINDING", GL_COPY_READ_BUFFER_BINDING),
      INST_ENUM("COPY_WRITE_BUFFER_BINDING", GL_COPY_WRITE_BUFFER_BINDING),
      INST_ENUM("UNIFORM_BUFFER", GL_UNIFORM_BUFFER),
      INST_ENUM("UNIFORM_BUFFER_BINDING", GL_UNIFORM_BUFFER_BINDING),
      INST_ENUM("UNIFORM_BUFFER_START", GL_UNIFORM_BUFFER_START),
      INST_ENUM("UNIFORM_BUFFER_SIZE", GL_UNIFORM_BUFFER_SIZE),
      INST_ENUM("MAX_VERTEX_UNIFORM_BLOCKS", GL_MAX_VERTEX_UNIFORM_BLOCKS),
      INST_ENUM("MAX_FRAGMENT_UNIFORM_BLOCKS", GL_MAX_FRAGMENT_UNIFORM_BLOCKS),
      INST_ENUM("MAX_COMBINED_UNIFORM_BLOCKS", GL_MAX_COMBINED_UNIFORM_BLOCKS),
      INST_ENUM("MAX_UNIFORM_BUFFER_BINDINGS", GL_MAX_UNIFORM_BUFFER_BINDINGS),
      INST_ENUM("MAX_UNIFORM_BLOCK_SIZE", GL_MAX_UNIFORM_BLOCK_SIZE),
      INST_ENUM("MAX_COMBINED_VERTEX_UNIFORM_COMPONENTS",
                GL_MAX_COMBINED_VERTEX_UNIFORM_COMPONENTS),
      INST_ENUM("MAX_COMBINED_FRAGMENT_UNIFORM_COMPONENTS",
                GL_MAX_COMBINED_FRAGMENT_UNIFORM_COMPONENTS),
      INST_ENUM("UNIFORM_BUFFER_OFFSET_ALIGNMENT", GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT),
      INST_ENUM("ACTIVE_UNIFORM_BLOCKS", GL_ACTIVE_UNIFORM_BLOCKS),
      INST_ENUM("UNIFORM_TYPE", GL_UNIFORM_TYPE),
      INST_ENUM("UNIFORM_SIZE", GL_UNIFORM_SIZE),
      INST_ENUM("UNIFORM_BLOCK_INDEX", GL_UNIFORM_BLOCK_INDEX),
      INST_ENUM("UNIFORM_OFFSET", GL_UNIFORM_OFFSET),
      INST_ENUM("UNIFORM_ARRAY_STRIDE", GL_UNIFORM_ARRAY_STRIDE),
      INST_ENUM("UNIFORM_MATRIX_STRIDE", GL_UNIFORM_MATRIX_STRIDE),
      INST_ENUM("UNIFORM_IS_ROW_MAJOR", GL_UNIFORM_IS_ROW_MAJOR),
      INST_ENUM("UNIFORM_BLOCK_BINDING", GL_UNIFORM_BLOCK_BINDING),
      INST_ENUM("UNIFORM_BLOCK_DATA_SIZE", GL_UNIFORM_BLOCK_DATA_SIZE),
      INST_ENUM("UNIFORM_BLOCK_ACTIVE_UNIFORMS", GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS),
      INST_ENUM("UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES", GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES),
      INST_ENUM("UNIFORM_BLOCK_REFERENCED_BY_VERTEX_SHADER",
                GL_UNIFORM_BLOCK_REFERENCED_BY_VERTEX_SHADER),
      INST_ENUM("UNIFORM_BLOCK_REFERENCED_BY_FRAGMENT_SHADER",
                GL_UNIFORM_BLOCK_REFERENCED_BY_FRAGMENT_SHADER),
      INST_ENUM("INVALID_INDEX", GL_INVALID_INDEX),
      INST_ENUM("MAX_VERTEX_OUTPUT_COMPONENTS", GL_MAX_VERTEX_OUTPUT_COMPONENTS),
      INST_ENUM("MAX_FRAGMENT_INPUT_COMPONENTS", GL_MAX_FRAGMENT_INPUT_COMPONENTS),
      INST_ENUM("MAX_SERVER_WAIT_TIMEOUT", GL_MAX_SERVER_WAIT_TIMEOUT),
      INST_ENUM("OBJECT_TYPE", GL_OBJECT_TYPE),
      INST_ENUM("SYNC_CONDITION", GL_SYNC_CONDITION),
      INST_ENUM("SYNC_STATUS", GL_SYNC_STATUS),
      INST_ENUM("SYNC_FLAGS", GL_SYNC_FLAGS),
      INST_ENUM("SYNC_FENCE", GL_SYNC_FENCE),
      INST_ENUM("SYNC_GPU_COMMANDS_COMPLETE", GL_SYNC_GPU_COMMANDS_COMPLETE),
      INST_ENUM("UNSIGNALED", GL_UNSIGNALED),
      INST_ENUM("SIGNALED", GL_SIGNALED),
      INST_ENUM("ALREADY_SIGNALED", GL_ALREADY_SIGNALED),
      INST_ENUM("TIMEOUT_EXPIRED", GL_TIMEOUT_EXPIRED),
      INST_ENUM("CONDITION_SATISFIED", GL_CONDITION_SATISFIED),
      INST_ENUM("WAIT_FAILED", GL_WAIT_FAILED),
      INST_ENUM("SYNC_FLUSH_COMMANDS_BIT", GL_SYNC_FLUSH_COMMANDS_BIT),
      INST_ENUM("VERTEX_ATTRIB_ARRAY_DIVISOR", GL_VERTEX_ATTRIB_ARRAY_DIVISOR),
      INST_ENUM("ANY_SAMPLES_PASSED", GL_ANY_SAMPLES_PASSED),
      INST_ENUM("ANY_SAMPLES_PASSED_CONSERVATIVE", GL_ANY_SAMPLES_PASSED_CONSERVATIVE),
      INST_ENUM("SAMPLER_BINDING", GL_SAMPLER_BINDING),
      INST_ENUM("RGB10_A2UI", GL_RGB10_A2UI),
      INST_ENUM("INT_2_10_10_10_REV", GL_INT_2_10_10_10_REV),
      INST_ENUM("TRANSFORM_FEEDBACK", GL_TRANSFORM_FEEDBACK),
      INST_ENUM("TRANSFORM_FEEDBACK_PAUSED", GL_TRANSFORM_FEEDBACK_PAUSED),
      INST_ENUM("TRANSFORM_FEEDBACK_ACTIVE", GL_TRANSFORM_FEEDBACK_ACTIVE),
      INST_ENUM("TRANSFORM_FEEDBACK_BINDING", GL_TRANSFORM_FEEDBACK_BINDING),
      INST_ENUM("TEXTURE_IMMUTABLE_FORMAT", GL_TEXTURE_IMMUTABLE_FORMAT),
      INST_ENUM("MAX_ELEMENT_INDEX", GL_MAX_ELEMENT_INDEX),
      INST_ENUM("TEXTURE_IMMUTABLE_LEVELS", GL_TEXTURE_IMMUTABLE_LEVELS),
      INST_ENUM("TIMEOUT_IGNORED", GL_TIMEOUT_IGNORED),
      INST_ENUM("MAX_CLIENT_WAIT_TIMEOUT_WEBGL", GL_MAX_CLIENT_WAIT_TIMEOUT_WEBGL),
      INST_ENUM("DEPTH_BUFFER_BIT", GL_DEPTH_BUFFER_BIT),
      INST_ENUM("STENCIL_BUFFER_BIT", GL_STENCIL_BUFFER_BIT),
      INST_ENUM("COLOR_BUFFER_BIT", GL_COLOR_BUFFER_BIT),
      INST_ENUM("POINTS", GL_POINTS),
      INST_ENUM("POINT_SPRITE", GL_POINT_SPRITE),
      INST_ENUM("PROGRAM_POINT_SIZE", GL_PROGRAM_POINT_SIZE),
      INST_ENUM("LINES", GL_LINES),
      INST_ENUM("LINE_LOOP", GL_LINE_LOOP),
      INST_ENUM("LINE_STRIP", GL_LINE_STRIP),
      INST_ENUM("TRIANGLES", GL_TRIANGLES),
      INST_ENUM("TRIANGLE_STRIP", GL_TRIANGLE_STRIP),
      INST_ENUM("TRIANGLE_FAN", GL_TRIANGLE_FAN),
      INST_ENUM("ZERO", GL_ZERO),
      INST_ENUM("ONE", GL_ONE),
      INST_ENUM("SRC_COLOR", GL_SRC_COLOR),
      INST_ENUM("ONE_MINUS_SRC_COLOR", GL_ONE_MINUS_SRC_COLOR),
      INST_ENUM("SRC_ALPHA", GL_SRC_ALPHA),
      INST_ENUM("ONE_MINUS_SRC_ALPHA", GL_ONE_MINUS_SRC_ALPHA),
      INST_ENUM("DST_ALPHA", GL_DST_ALPHA),
      INST_ENUM("ONE_MINUS_DST_ALPHA", GL_ONE_MINUS_DST_ALPHA),
      INST_ENUM("DST_COLOR", GL_DST_COLOR),
      INST_ENUM("ONE_MINUS_DST_COLOR", GL_ONE_MINUS_DST_COLOR),
      INST_ENUM("SRC_ALPHA_SATURATE", GL_SRC_ALPHA_SATURATE),
      INST_ENUM("FUNC_ADD", GL_FUNC_ADD),
      INST_ENUM("BLEND_EQUATION", GL_BLEND_EQUATION),
      INST_ENUM("BLEND_EQUATION_RGB", GL_BLEND_EQUATION_RGB),
      INST_ENUM("BLEND_EQUATION_ALPHA", GL_BLEND_EQUATION_ALPHA),
      INST_ENUM("FUNC_SUBTRACT", GL_FUNC_SUBTRACT),
      INST_ENUM("FUNC_REVERSE_SUBTRACT", GL_FUNC_REVERSE_SUBTRACT),
      INST_ENUM("BLEND_DST_RGB", GL_BLEND_DST_RGB),
      INST_ENUM("BLEND_SRC_RGB", GL_BLEND_SRC_RGB),
      INST_ENUM("BLEND_DST_ALPHA", GL_BLEND_DST_ALPHA),
      INST_ENUM("BLEND_SRC_ALPHA", GL_BLEND_SRC_ALPHA),
      INST_ENUM("CONSTANT_COLOR", GL_CONSTANT_COLOR),
      INST_ENUM("ONE_MINUS_CONSTANT_COLOR", GL_ONE_MINUS_CONSTANT_COLOR),
      INST_ENUM("CONSTANT_ALPHA", GL_CONSTANT_ALPHA),
      INST_ENUM("ONE_MINUS_CONSTANT_ALPHA", GL_ONE_MINUS_CONSTANT_ALPHA),
      INST_ENUM("BLEND_COLOR", GL_BLEND_COLOR),
      INST_ENUM("ARRAY_BUFFER", GL_ARRAY_BUFFER),
      INST_ENUM("ELEMENT_ARRAY_BUFFER", GL_ELEMENT_ARRAY_BUFFER),
      INST_ENUM("ARRAY_BUFFER_BINDING", GL_ARRAY_BUFFER_BINDING),
      INST_ENUM("ELEMENT_ARRAY_BUFFER_BINDING", GL_ELEMENT_ARRAY_BUFFER_BINDING),
      INST_ENUM("STREAM_DRAW", GL_STREAM_DRAW),
      INST_ENUM("STATIC_DRAW", GL_STATIC_DRAW),
      INST_ENUM("DYNAMIC_DRAW", GL_DYNAMIC_DRAW),
      INST_ENUM("BUFFER_SIZE", GL_BUFFER_SIZE),
      INST_ENUM("BUFFER_USAGE", GL_BUFFER_USAGE),
      INST_ENUM("CURRENT_VERTEX_ATTRIB", GL_CURRENT_VERTEX_ATTRIB),
      INST_ENUM("FRONT", GL_FRONT),
      INST_ENUM("BACK", GL_BACK),
      INST_ENUM("FRONT_AND_BACK", GL_FRONT_AND_BACK),
      INST_ENUM("TEXTURE_2D", GL_TEXTURE_2D),
      INST_ENUM("CULL_FACE", GL_CULL_FACE),
      INST_ENUM("BLEND", GL_BLEND),
      INST_ENUM("DITHER", GL_DITHER),
      INST_ENUM("STENCIL_TEST", GL_STENCIL_TEST),
      INST_ENUM("DEPTH_TEST", GL_DEPTH_TEST),
      INST_ENUM("SCISSOR_TEST", GL_SCISSOR_TEST),
      INST_ENUM("POLYGON_OFFSET_FILL", GL_POLYGON_OFFSET_FILL),
      INST_ENUM("SAMPLE_ALPHA_TO_COVERAGE", GL_SAMPLE_ALPHA_TO_COVERAGE),
      INST_ENUM("SAMPLE_COVERAGE", GL_SAMPLE_COVERAGE),
      INST_ENUM("NO_ERROR", GL_NO_ERROR),
      INST_ENUM("INVALID_ENUM", GL_INVALID_ENUM),
      INST_ENUM("INVALID_VALUE", GL_INVALID_VALUE),
      INST_ENUM("INVALID_OPERATION", GL_INVALID_OPERATION),
      INST_ENUM("OUT_OF_MEMORY", GL_OUT_OF_MEMORY),
      INST_ENUM("CW", GL_CW),
      INST_ENUM("CCW", GL_CCW),
      INST_ENUM("LINE_WIDTH", GL_LINE_WIDTH),
      INST_ENUM("ALIASED_POINT_SIZE_RANGE", GL_ALIASED_POINT_SIZE_RANGE),
      INST_ENUM("ALIASED_LINE_WIDTH_RANGE", GL_ALIASED_LINE_WIDTH_RANGE),
      INST_ENUM("CULL_FACE_MODE", GL_CULL_FACE_MODE),
      INST_ENUM("FRONT_FACE", GL_FRONT_FACE),
      INST_ENUM("DEPTH_RANGE", GL_DEPTH_RANGE),
      INST_ENUM("DEPTH_WRITEMASK", GL_DEPTH_WRITEMASK),
      INST_ENUM("DEPTH_CLEAR_VALUE", GL_DEPTH_CLEAR_VALUE),
      INST_ENUM("DEPTH_FUNC", GL_DEPTH_FUNC),
      INST_ENUM("STENCIL_CLEAR_VALUE", GL_STENCIL_CLEAR_VALUE),
      INST_ENUM("STENCIL_FUNC", GL_STENCIL_FUNC),
      INST_ENUM("STENCIL_FAIL", GL_STENCIL_FAIL),
      INST_ENUM("STENCIL_PASS_DEPTH_FAIL", GL_STENCIL_PASS_DEPTH_FAIL),
      INST_ENUM("STENCIL_PASS_DEPTH_PASS", GL_STENCIL_PASS_DEPTH_PASS),
      INST_ENUM("STENCIL_REF", GL_STENCIL_REF),
      INST_ENUM("STENCIL_VALUE_MASK", GL_STENCIL_VALUE_MASK),
      INST_ENUM("STENCIL_WRITEMASK", GL_STENCIL_WRITEMASK),
      INST_ENUM("STENCIL_BACK_FUNC", GL_STENCIL_BACK_FUNC),
      INST_ENUM("STENCIL_BACK_FAIL", GL_STENCIL_BACK_FAIL),
      INST_ENUM("STENCIL_BACK_PASS_DEPTH_FAIL", GL_STENCIL_BACK_PASS_DEPTH_FAIL),
      INST_ENUM("STENCIL_BACK_PASS_DEPTH_PASS", GL_STENCIL_BACK_PASS_DEPTH_PASS),
      INST_ENUM("STENCIL_BACK_REF", GL_STENCIL_BACK_REF),
      INST_ENUM("STENCIL_BACK_VALUE_MASK", GL_STENCIL_BACK_VALUE_MASK),
      INST_ENUM("STENCIL_BACK_WRITEMASK", GL_STENCIL_BACK_WRITEMASK),
      INST_ENUM("VIEWPORT", GL_VIEWPORT),
      INST_ENUM("SCISSOR_BOX", GL_SCISSOR_BOX),
      INST_ENUM("COLOR_CLEAR_VALUE", GL_COLOR_CLEAR_VALUE),
      INST_ENUM("COLOR_WRITEMASK", GL_COLOR_WRITEMASK),
      INST_ENUM("UNPACK_ALIGNMENT", GL_UNPACK_ALIGNMENT),
      INST_ENUM("PACK_ALIGNMENT", GL_PACK_ALIGNMENT),
      INST_ENUM("MAX_TEXTURE_SIZE", GL_MAX_TEXTURE_SIZE),
      INST_ENUM("MAX_VIEWPORT_DIMS", GL_MAX_VIEWPORT_DIMS),
      INST_ENUM("SUBPIXEL_BITS", GL_SUBPIXEL_BITS),
      INST_ENUM("RED_BITS", GL_RED_BITS),
      INST_ENUM("GREEN_BITS", GL_GREEN_BITS),
      INST_ENUM("BLUE_BITS", GL_BLUE_BITS),
      INST_ENUM("ALPHA_BITS", GL_ALPHA_BITS),
      INST_ENUM("DEPTH_BITS", GL_DEPTH_BITS),
      INST_ENUM("STENCIL_BITS", GL_STENCIL_BITS),
      INST_ENUM("POLYGON_OFFSET_UNITS", GL_POLYGON_OFFSET_UNITS),
      INST_ENUM("POLYGON_OFFSET_FACTOR", GL_POLYGON_OFFSET_FACTOR),
      INST_ENUM("TEXTURE_BINDING_2D", GL_TEXTURE_BINDING_2D),
      INST_ENUM("SAMPLE_BUFFERS", GL_SAMPLE_BUFFERS),
      INST_ENUM("SAMPLES", GL_SAMPLES),
      INST_ENUM("SAMPLE_COVERAGE_VALUE", GL_SAMPLE_COVERAGE_VALUE),
      INST_ENUM("SAMPLE_COVERAGE_INVERT", GL_SAMPLE_COVERAGE_INVERT),
      INST_ENUM("COMPRESSED_TEXTURE_FORMATS", GL_COMPRESSED_TEXTURE_FORMATS),
      INST_ENUM("DONT_CARE", GL_DONT_CARE),
      INST_ENUM("FASTEST", GL_FASTEST),
      INST_ENUM("NICEST", GL_NICEST),
      INST_ENUM("GENERATE_MIPMAP_HINT", GL_GENERATE_MIPMAP_HINT),
      INST_ENUM("BYTE", GL_BYTE),
      INST_ENUM("UNSIGNED_BYTE", GL_UNSIGNED_BYTE),
      INST_ENUM("SHORT", GL_SHORT),
      INST_ENUM("UNSIGNED_SHORT", GL_UNSIGNED_SHORT),
      INST_ENUM("INT", GL_INT),
      INST_ENUM("UNSIGNED_INT", GL_UNSIGNED_INT),
      INST_ENUM("FLOAT", GL_FLOAT),
      INST_ENUM("DEPTH_COMPONENT", GL_DEPTH_COMPONENT),
      INST_ENUM("ALPHA", GL_ALPHA),
      INST_ENUM("RGB", GL_RGB),
      INST_ENUM("RGBA", GL_RGBA),
      INST_ENUM("LUMINANCE", GL_LUMINANCE),
      INST_ENUM("LUMINANCE_ALPHA", GL_LUMINANCE_ALPHA),
      INST_ENUM("UNSIGNED_SHORT_4_4_4_4", GL_UNSIGNED_SHORT_4_4_4_4),
      INST_ENUM("UNSIGNED_SHORT_5_5_5_1", GL_UNSIGNED_SHORT_5_5_5_1),
      INST_ENUM("UNSIGNED_SHORT_5_6_5", GL_UNSIGNED_SHORT_5_6_5),
      INST_ENUM("FRAGMENT_SHADER", GL_FRAGMENT_SHADER),
      INST_ENUM("VERTEX_SHADER", GL_VERTEX_SHADER),
      INST_ENUM("MAX_VERTEX_ATTRIBS", GL_MAX_VERTEX_ATTRIBS),
      INST_ENUM("MAX_VERTEX_UNIFORM_VECTORS", GL_MAX_VERTEX_UNIFORM_VECTORS),
      INST_ENUM("MAX_VARYING_VECTORS", GL_MAX_VARYING_VECTORS),
      INST_ENUM("MAX_COMBINED_TEXTURE_IMAGE_UNITS", GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS),
      INST_ENUM("MAX_VERTEX_TEXTURE_IMAGE_UNITS", GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS),
      INST_ENUM("MAX_TEXTURE_IMAGE_UNITS", GL_MAX_TEXTURE_IMAGE_UNITS),
      INST_ENUM("MAX_FRAGMENT_UNIFORM_VECTORS", GL_MAX_FRAGMENT_UNIFORM_VECTORS),
      INST_ENUM("SHADER_TYPE", GL_SHADER_TYPE),
      INST_ENUM("DELETE_STATUS", GL_DELETE_STATUS),
      INST_ENUM("LINK_STATUS", GL_LINK_STATUS),
      INST_ENUM("VALIDATE_STATUS", GL_VALIDATE_STATUS),
      INST_ENUM("ATTACHED_SHADERS", GL_ATTACHED_SHADERS),
      INST_ENUM("ACTIVE_UNIFORMS", GL_ACTIVE_UNIFORMS),
      INST_ENUM("ACTIVE_ATTRIBUTES", GL_ACTIVE_ATTRIBUTES),
      INST_ENUM("SHADING_LANGUAGE_VERSION", GL_SHADING_LANGUAGE_VERSION),
      INST_ENUM("CURRENT_PROGRAM", GL_CURRENT_PROGRAM),
      INST_ENUM("NEVER", GL_NEVER),
      INST_ENUM("LESS", GL_LESS),
      INST_ENUM("EQUAL", GL_EQUAL),
      INST_ENUM("LEQUAL", GL_LEQUAL),
      INST_ENUM("GREATER", GL_GREATER),
      INST_ENUM("NOTEQUAL", GL_NOTEQUAL),
      INST_ENUM("GEQUAL", GL_GEQUAL),
      INST_ENUM("ALWAYS", GL_ALWAYS),
      INST_ENUM("KEEP", GL_KEEP),
      INST_ENUM("REPLACE", GL_REPLACE),
      INST_ENUM("INCR", GL_INCR),
      INST_ENUM("DECR", GL_DECR),
      INST_ENUM("INVERT", GL_INVERT),
      INST_ENUM("INCR_WRAP", GL_INCR_WRAP),
      INST_ENUM("DECR_WRAP", GL_DECR_WRAP),
      INST_ENUM("VENDOR", GL_VENDOR),
      INST_ENUM("RENDERER", GL_RENDERER),
      INST_ENUM("VERSION", GL_VERSION),
      INST_ENUM("NEAREST", GL_NEAREST),
      INST_ENUM("LINEAR", GL_LINEAR),
      INST_ENUM("NEAREST_MIPMAP_NEAREST", GL_NEAREST_MIPMAP_NEAREST),
      INST_ENUM("LINEAR_MIPMAP_NEAREST", GL_LINEAR_MIPMAP_NEAREST),
      INST_ENUM("NEAREST_MIPMAP_LINEAR", GL_NEAREST_MIPMAP_LINEAR),
      INST_ENUM("LINEAR_MIPMAP_LINEAR", GL_LINEAR_MIPMAP_LINEAR),
      INST_ENUM("TEXTURE_MAG_FILTER", GL_TEXTURE_MAG_FILTER),
      INST_ENUM("TEXTURE_MIN_FILTER", GL_TEXTURE_MIN_FILTER),
      INST_ENUM("TEXTURE_WRAP_S", GL_TEXTURE_WRAP_S),
      INST_ENUM("TEXTURE_WRAP_T", GL_TEXTURE_WRAP_T),
      INST_ENUM("TEXTURE", GL_TEXTURE),
      INST_ENUM("TEXTURE_CUBE_MAP", GL_TEXTURE_CUBE_MAP),
      INST_ENUM("TEXTURE_BINDING_CUBE_MAP", GL_TEXTURE_BINDING_CUBE_MAP),
      INST_ENUM("TEXTURE_CUBE_MAP_POSITIVE_X", GL_TEXTURE_CUBE_MAP_POSITIVE_X),
      INST_ENUM("TEXTURE_CUBE_MAP_NEGATIVE_X", GL_TEXTURE_CUBE_MAP_NEGATIVE_X),
      INST_ENUM("TEXTURE_CUBE_MAP_POSITIVE_Y", GL_TEXTURE_CUBE_MAP_POSITIVE_Y),
      INST_ENUM("TEXTURE_CUBE_MAP_NEGATIVE_Y", GL_TEXTURE_CUBE_MAP_NEGATIVE_Y),
      INST_ENUM("TEXTURE_CUBE_MAP_POSITIVE_Z", GL_TEXTURE_CUBE_MAP_POSITIVE_Z),
      INST_ENUM("TEXTURE_CUBE_MAP_NEGATIVE_Z", GL_TEXTURE_CUBE_MAP_NEGATIVE_Z),
      INST_ENUM("MAX_CUBE_MAP_TEXTURE_SIZE", GL_MAX_CUBE_MAP_TEXTURE_SIZE),
      INST_ENUM("TEXTURE0", GL_TEXTURE0),
      INST_ENUM("TEXTURE1", GL_TEXTURE1),
      INST_ENUM("TEXTURE2", GL_TEXTURE2),
      INST_ENUM("TEXTURE3", GL_TEXTURE3),
      INST_ENUM("TEXTURE4", GL_TEXTURE4),
      INST_ENUM("TEXTURE5", GL_TEXTURE5),
      INST_ENUM("TEXTURE6", GL_TEXTURE6),
      INST_ENUM("TEXTURE7", GL_TEXTURE7),
      INST_ENUM("TEXTURE8", GL_TEXTURE8),
      INST_ENUM("TEXTURE9", GL_TEXTURE9),
      INST_ENUM("TEXTURE10", GL_TEXTURE10),
      INST_ENUM("TEXTURE11", GL_TEXTURE11),
      INST_ENUM("TEXTURE12", GL_TEXTURE12),
      INST_ENUM("TEXTURE13", GL_TEXTURE13),
      INST_ENUM("TEXTURE14", GL_TEXTURE14),
      INST_ENUM("TEXTURE15", GL_TEXTURE15),
      INST_ENUM("TEXTURE16", GL_TEXTURE16),
      INST_ENUM("TEXTURE17", GL_TEXTURE17),
      INST_ENUM("TEXTURE18", GL_TEXTURE18),
      INST_ENUM("TEXTURE19", GL_TEXTURE19),
      INST_ENUM("TEXTURE20", GL_TEXTURE20),
      INST_ENUM("TEXTURE21", GL_TEXTURE21),
      INST_ENUM("TEXTURE22", GL_TEXTURE22),
      INST_ENUM("TEXTURE23", GL_TEXTURE23),
      INST_ENUM("TEXTURE24", GL_TEXTURE24),
      INST_ENUM("TEXTURE25", GL_TEXTURE25),
      INST_ENUM("TEXTURE26", GL_TEXTURE26),
      INST_ENUM("TEXTURE27", GL_TEXTURE27),
      INST_ENUM("TEXTURE28", GL_TEXTURE28),
      INST_ENUM("TEXTURE29", GL_TEXTURE29),
      INST_ENUM("TEXTURE30", GL_TEXTURE30),
      INST_ENUM("TEXTURE31", GL_TEXTURE31),
      INST_ENUM("ACTIVE_TEXTURE", GL_ACTIVE_TEXTURE),
      INST_ENUM("REPEAT", GL_REPEAT),
      INST_ENUM("CLAMP_TO_EDGE", GL_CLAMP_TO_EDGE),
      INST_ENUM("MIRRORED_REPEAT", GL_MIRRORED_REPEAT),
      INST_ENUM("FLOAT_VEC2", GL_FLOAT_VEC2),
      INST_ENUM("FLOAT_VEC3", GL_FLOAT_VEC3),
      INST_ENUM("FLOAT_VEC4", GL_FLOAT_VEC4),
      INST_ENUM("INT_VEC2", GL_INT_VEC2),
      INST_ENUM("INT_VEC3", GL_INT_VEC3),
      INST_ENUM("INT_VEC4", GL_INT_VEC4),
      INST_ENUM("BOOL", GL_BOOL),
      INST_ENUM("BOOL_VEC2", GL_BOOL_VEC2),
      INST_ENUM("BOOL_VEC3", GL_BOOL_VEC3),
      INST_ENUM("BOOL_VEC4", GL_BOOL_VEC4),
      INST_ENUM("FLOAT_MAT2", GL_FLOAT_MAT2),
      INST_ENUM("FLOAT_MAT3", GL_FLOAT_MAT3),
      INST_ENUM("FLOAT_MAT4", GL_FLOAT_MAT4),
      INST_ENUM("SAMPLER_2D", GL_SAMPLER_2D),
      INST_ENUM("SAMPLER_CUBE", GL_SAMPLER_CUBE),
      INST_ENUM("VERTEX_ATTRIB_ARRAY_ENABLED", GL_VERTEX_ATTRIB_ARRAY_ENABLED),
      INST_ENUM("VERTEX_ATTRIB_ARRAY_SIZE", GL_VERTEX_ATTRIB_ARRAY_SIZE),
      INST_ENUM("VERTEX_ATTRIB_ARRAY_STRIDE", GL_VERTEX_ATTRIB_ARRAY_STRIDE),
      INST_ENUM("VERTEX_ATTRIB_ARRAY_TYPE", GL_VERTEX_ATTRIB_ARRAY_TYPE),
      INST_ENUM("VERTEX_ATTRIB_ARRAY_NORMALIZED", GL_VERTEX_ATTRIB_ARRAY_NORMALIZED),
      INST_ENUM("VERTEX_ATTRIB_ARRAY_POINTER", GL_VERTEX_ATTRIB_ARRAY_POINTER),
      INST_ENUM("VERTEX_ATTRIB_ARRAY_BUFFER_BINDING", GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING),
      INST_ENUM("IMPLEMENTATION_COLOR_READ_TYPE", GL_IMPLEMENTATION_COLOR_READ_TYPE),
      INST_ENUM("IMPLEMENTATION_COLOR_READ_FORMAT", GL_IMPLEMENTATION_COLOR_READ_FORMAT),
      INST_ENUM("COMPILE_STATUS", GL_COMPILE_STATUS),
      INST_ENUM("LOW_FLOAT", GL_LOW_FLOAT),
      INST_ENUM("MEDIUM_FLOAT", GL_MEDIUM_FLOAT),
      INST_ENUM("HIGH_FLOAT", GL_HIGH_FLOAT),
      INST_ENUM("LOW_INT", GL_LOW_INT),
      INST_ENUM("MEDIUM_INT", GL_MEDIUM_INT),
      INST_ENUM("HIGH_INT", GL_HIGH_INT),
      INST_ENUM("FRAMEBUFFER", GL_FRAMEBUFFER),
      INST_ENUM("RENDERBUFFER", GL_RENDERBUFFER),
      INST_ENUM("RGBA4", GL_RGBA4),
      INST_ENUM("RGB5_A1", GL_RGB5_A1),
      INST_ENUM("RGB565", GL_RGB565),
      INST_ENUM("DEPTH_COMPONENT16", GL_DEPTH_COMPONENT16),
      INST_ENUM("STENCIL_INDEX8", GL_STENCIL_INDEX8),
      INST_ENUM("DEPTH_STENCIL", GL_DEPTH_STENCIL),
      INST_ENUM("RENDERBUFFER_WIDTH", GL_RENDERBUFFER_WIDTH),
      INST_ENUM("RENDERBUFFER_HEIGHT", GL_RENDERBUFFER_HEIGHT),
      INST_ENUM("RENDERBUFFER_INTERNAL_FORMAT", GL_RENDERBUFFER_INTERNAL_FORMAT),
      INST_ENUM("RENDERBUFFER_RED_SIZE", GL_RENDERBUFFER_RED_SIZE),
      INST_ENUM("RENDERBUFFER_GREEN_SIZE", GL_RENDERBUFFER_GREEN_SIZE),
      INST_ENUM("RENDERBUFFER_BLUE_SIZE", GL_RENDERBUFFER_BLUE_SIZE),
      INST_ENUM("RENDERBUFFER_ALPHA_SIZE", GL_RENDERBUFFER_ALPHA_SIZE),
      INST_ENUM("RENDERBUFFER_DEPTH_SIZE", GL_RENDERBUFFER_DEPTH_SIZE),
      INST_ENUM("RENDERBUFFER_STENCIL_SIZE", GL_RENDERBUFFER_STENCIL_SIZE),
      INST_ENUM("FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE", GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE),
      INST_ENUM("FRAMEBUFFER_ATTACHMENT_OBJECT_NAME", GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME),
      INST_ENUM("FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL", GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL),
      INST_ENUM("FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE",
                GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE),
      INST_ENUM("COLOR_ATTACHMENT0", GL_COLOR_ATTACHMENT0),
      INST_ENUM("DEPTH_ATTACHMENT", GL_DEPTH_ATTACHMENT),
      INST_ENUM("STENCIL_ATTACHMENT", GL_STENCIL_ATTACHMENT),
      INST_ENUM("DEPTH_STENCIL_ATTACHMENT", GL_DEPTH_STENCIL_ATTACHMENT),
      INST_ENUM("NONE", GL_NONE),
      INST_ENUM("FRAMEBUFFER_COMPLETE", GL_FRAMEBUFFER_COMPLETE),
      INST_ENUM("FRAMEBUFFER_INCOMPLETE_ATTACHMENT", GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT),
      INST_ENUM("FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT",
                GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT),
      INST_ENUM("FRAMEBUFFER_INCOMPLETE_DIMENSIONS", GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT),
      INST_ENUM("FRAMEBUFFER_UNSUPPORTED", GL_FRAMEBUFFER_UNSUPPORTED),
      INST_ENUM("FRAMEBUFFER_BINDING", GL_FRAMEBUFFER_BINDING),
      INST_ENUM("RENDERBUFFER_BINDING", GL_RENDERBUFFER_BINDING),
      INST_ENUM("MAX_RENDERBUFFER_SIZE", GL_MAX_RENDERBUFFER_SIZE),
      INST_ENUM("INVALID_FRAMEBUFFER_OPERATION", GL_INVALID_FRAMEBUFFER_OPERATION),
      INST_ENUM("UNPACK_FLIP_Y_WEBGL", GL_UNPACK_FLIP_Y_WEBGL),
      INST_ENUM("UNPACK_PREMULTIPLY_ALPHA_WEBGL", GL_UNPACK_PREMULTIPLY_ALPHA_WEBGL),
      INST_ENUM("CONTEXT_LOST_WEBGL", GL_CONTEXT_LOST_WEBGL),
      INST_ENUM("UNPACK_COLORSPACE_CONVERSION_WEBGL", GL_UNPACK_COLORSPACE_CONVERSION_WEBGL),
      INST_ENUM("BROWSER_DEFAULT_WEBGL", GL_BROWSER_DEFAULT_WEBGL),
      INST_ENUM("UNMASKED_VENDOR_WEBGL", GL_UNMASKED_VENDOR_WEBGL),
      INST_ENUM("UNMASKED_RENDERER_WEBGL", GL_UNMASKED_RENDERER_WEBGL),

#undef INST_ENUM
    });

  WebGL2RenderingContext::constructor = Napi::Persistent(ctor);
  WebGL2RenderingContext::constructor.SuppressDestruct();

  EXPORT_PROP(exports, "WebGL2RenderingContext", ctor);

  return exports;
};

// GL_EXPORT void glClear (GLbitfield mask);
Napi::Value WebGL2RenderingContext::Clear(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLbitfield mask   = args[0];
  clear_mask_ |= mask;
  GL_EXPORT::glClear(mask);
  return info.Env().Undefined();
}

// GL_EXPORT void glClearColor (GLclampf red, GLclampf green, GLclampf blue, GLclampf alpha);
Napi::Value WebGL2RenderingContext::ClearColor(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glClearColor(args[0], args[1], args[2], args[3]);
  return info.Env().Undefined();
}

// GL_EXPORT void glClearDepth (GLclampd depth);
Napi::Value WebGL2RenderingContext::ClearDepth(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glClearDepth(args[0]);
  return info.Env().Undefined();
}

// GL_EXPORT void glColorMask (GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha);
Napi::Value WebGL2RenderingContext::ColorMask(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glColorMask(args[0], args[1], args[2], args[3]);
  return info.Env().Undefined();
}

// GL_EXPORT void glCullFace (GLenum mode);
Napi::Value WebGL2RenderingContext::CullFace(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glCullFace(args[0]);
  return info.Env().Undefined();
}

// GL_EXPORT void glDepthFunc (GLenum func);
Napi::Value WebGL2RenderingContext::DepthFunc(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glDepthFunc(args[0]);
  return info.Env().Undefined();
}

// GL_EXPORT void glDepthMask (GLboolean flag);
Napi::Value WebGL2RenderingContext::DepthMask(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glDepthMask(args[0]);
  return info.Env().Undefined();
}

// GL_EXPORT void glDepthRange (GLclampd zNear, GLclampd zFar);
Napi::Value WebGL2RenderingContext::DepthRange(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glDepthRange(args[0], args[1]);
  return info.Env().Undefined();
}

// GL_EXPORT void glDisable (GLenum cap);
Napi::Value WebGL2RenderingContext::Disable(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glDisable(args[0]);
  return info.Env().Undefined();
}

// GL_EXPORT void glDrawArrays (GLenum mode, GLint first, GLsizei count);
Napi::Value WebGL2RenderingContext::DrawArrays(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glDrawArrays(args[0], args[1], args[2]);
  return info.Env().Undefined();
}

// GL_EXPORT void glDrawElements (GLenum mode, GLsizei count, GLenum type, const void *indices);
Napi::Value WebGL2RenderingContext::DrawElements(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glDrawElements(args[0], args[1], args[2], args[3]);
  return info.Env().Undefined();
}

// GL_EXPORT void glEnable (GLenum cap);
Napi::Value WebGL2RenderingContext::Enable(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glEnable(args[0]);
  return info.Env().Undefined();
}

// GL_EXPORT void glFinish (void);
Napi::Value WebGL2RenderingContext::Finish(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glFinish();
  return info.Env().Undefined();
}

// GL_EXPORT void glFlush (void);
Napi::Value WebGL2RenderingContext::Flush(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glFlush();
  return info.Env().Undefined();
}

// GL_EXPORT void glFrontFace (GLenum mode);
Napi::Value WebGL2RenderingContext::FrontFace(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glFrontFace(args[0]);
  return info.Env().Undefined();
}

// GL_EXPORT GLenum glGetError (void);
Napi::Value WebGL2RenderingContext::GetError(Napi::CallbackInfo const& info) {
  return CPPToNapi(info.Env())(GL_EXPORT::glGetError());
}

// GL_EXPORT void glGetParameter (GLint pname);
Napi::Value WebGL2RenderingContext::GetParameter(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint pname       = args[0];
  switch (pname) {
    case GL_GPU_DISJOINT:
    case GL_MAX_CLIENT_WAIT_TIMEOUT_WEBGL: return CPPToNapi(info)(0);

    case GL_VENDOR:
    case GL_UNMASKED_VENDOR_WEBGL: {
      auto str = GL_EXPORT::glGetString(GL_VENDOR);
      if (str == NULL) { return info.Env().Null(); }
      return CPPToNapi(info)(std::string{reinterpret_cast<const GLchar*>(str)});
    }
    case GL_RENDERER:
    case GL_UNMASKED_RENDERER_WEBGL: {
      auto str = GL_EXPORT::glGetString(GL_RENDERER);
      if (str == NULL) { return info.Env().Null(); }
      return CPPToNapi(info)(std::string{reinterpret_cast<const GLchar*>(str)});
    }

    case GL_BLEND:
    case GL_CULL_FACE:
    case GL_DEPTH_TEST:
    case GL_DEPTH_WRITEMASK:
    case GL_DITHER:
    case GL_POLYGON_OFFSET_FILL:
    case GL_SAMPLE_COVERAGE_INVERT:
    case GL_SCISSOR_TEST:
    case GL_STENCIL_TEST:
    case GL_UNPACK_FLIP_Y_WEBGL:
    case GL_UNPACK_PREMULTIPLY_ALPHA_WEBGL: {
      GLubyte param{};
      GL_EXPORT::glGetBooleanv(pname, &param);
      return CPPToNapi(info)(static_cast<bool>(param));
    }

    case GL_COLOR_WRITEMASK: {
      std::vector<GLboolean> params(4);
      GL_EXPORT::glGetBooleanv(pname, params.data());
      return CPPToNapi(info)(std::vector<bool>{params.begin(), params.end()});
    }

    case GL_ARRAY_BUFFER_BINDING:
    case GL_CURRENT_PROGRAM:
    case GL_ELEMENT_ARRAY_BUFFER_BINDING:
    case GL_FRAMEBUFFER_BINDING:
    case GL_RENDERBUFFER_BINDING:
    case GL_TEXTURE_BINDING_2D:
    case GL_TEXTURE_BINDING_CUBE_MAP: {
      GLint params{};
      GL_EXPORT::glGetIntegerv(pname, &params);
      return CPPToNapi(info)(params);
    }

    case GL_DEPTH_CLEAR_VALUE:
    case GL_LINE_WIDTH:
    case GL_POLYGON_OFFSET_FACTOR:
    case GL_POLYGON_OFFSET_UNITS:
    case GL_SAMPLE_COVERAGE_VALUE: {
      GLfloat param{};
      GL_EXPORT::glGetFloatv(pname, &param);
      return CPPToNapi(info)(param);
    }

    case GL_SHADING_LANGUAGE_VERSION:
    case GL_VERSION:
    case GL_EXTENSIONS: {
      auto str = GL_EXPORT::glGetString(pname);
      if (str == NULL) { return info.Env().Null(); }
      return CPPToNapi(info)(std::string{reinterpret_cast<const GLchar*>(str)});
    }

    case GL_MAX_VIEWPORT_DIMS: {
      std::vector<GLint> params(2);
      GL_EXPORT::glGetIntegerv(pname, params.data());
      return CPPToNapi(info)(params);
    }

    case GL_VIEWPORT:
    case GL_SCISSOR_BOX: {
      std::vector<GLint> params(4);
      GL_EXPORT::glGetIntegerv(pname, params.data());
      return CPPToNapi(info)(params);
    }

    case GL_DEPTH_RANGE:
    case GL_ALIASED_LINE_WIDTH_RANGE:
    case GL_ALIASED_POINT_SIZE_RANGE: {
      std::vector<GLfloat> params(2);
      GL_EXPORT::glGetFloatv(pname, params.data());
      return CPPToNapi(info)(params);
    }

    case GL_BLEND_COLOR:
    case GL_COLOR_CLEAR_VALUE: {
      std::vector<GLfloat> params(4);
      GL_EXPORT::glGetFloatv(pname, params.data());
      return CPPToNapi(info)(params);
    }

    default: {
      GLint params{};
      GL_EXPORT::glGetIntegerv(pname, &params);
      return CPPToNapi(info)(params);
    }
  }
}

// GL_EXPORT const GLubyte * glGetString (GL_EXTENSIONS);
Napi::Value WebGL2RenderingContext::GetSupportedExtensions(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  auto str          = reinterpret_cast<const GLchar*>(GL_EXPORT::glGetString(GL_EXTENSIONS));
  if (str == NULL) { return info.Env().Null(); }
  auto iss   = std::istringstream{str};
  auto begin = std::istream_iterator<std::string>{iss};
  auto end   = std::istream_iterator<std::string>{};
  std::vector<std::string> extensions{begin, end};
  return CPPToNapi(info)(extensions);
}

// GL_EXPORT void glHint (GLenum target, GLenum mode);
Napi::Value WebGL2RenderingContext::Hint(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glHint(args[0], args[1]);
  return info.Env().Undefined();
}

// GL_EXPORT GLboolean glIsEnabled (GLenum cap);
Napi::Value WebGL2RenderingContext::IsEnabled(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  auto enabled      = GL_EXPORT::glIsEnabled(args[0]);
  return CPPToNapi(info.Env())(enabled);
}

// GL_EXPORT void glLineWidth (GLfloat width);
Napi::Value WebGL2RenderingContext::LineWidth(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glLineWidth(args[0]);
  return info.Env().Undefined();
}

// GL_EXPORT void glPixelStorei (GLenum pname, GLint param);
Napi::Value WebGL2RenderingContext::PixelStorei(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLuint pname      = args[0];
  switch (pname) {
    case GL_PACK_ALIGNMENT:
    case GL_UNPACK_ALIGNMENT: GL_EXPORT::glPixelStorei(pname, args[1]);
    case GL_UNPACK_FLIP_Y_WEBGL:
    case GL_UNPACK_PREMULTIPLY_ALPHA_WEBGL:
    default: break;
  }
  return info.Env().Undefined();
}

// GL_EXPORT void glPolygonOffset (GLfloat factor, GLfloat units);
Napi::Value WebGL2RenderingContext::PolygonOffset(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glPolygonOffset(args[0], args[1]);
  return info.Env().Undefined();
}

// GL_EXPORT void glReadPixels (GLint x, GLint y, GLsizei width, GLsizei height, GLenum format,
// GLenum type, void *pixels);
Napi::Value WebGL2RenderingContext::ReadPixels(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint x           = args[0];
  GLint y           = args[1];
  GLsizei width     = args[2];
  GLsizei height    = args[3];
  GLenum format     = args[4];
  GLenum type       = args[5];
  if (!info[6].IsNumber()) {
    Span<char> ptr = args[6];
    GL_EXPORT::glReadnPixels(x, y, width, height, format, type, ptr.size(), ptr.data());
  } else {
    GLintptr ptr = args[6];
    GL_EXPORT::glReadPixels(x, y, width, height, format, type, reinterpret_cast<void*>(ptr));
  }
  return info.Env().Undefined();
}

// GL_EXPORT void glScissor (GLint x, GLint y, GLsizei width, GLsizei height);
Napi::Value WebGL2RenderingContext::Scissor(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glScissor(args[0], args[1], args[2], args[3]);
  return info.Env().Undefined();
}

// GL_EXPORT void glViewport (GLint x, GLint y, GLsizei width, GLsizei height);
Napi::Value WebGL2RenderingContext::Viewport(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glViewport(args[0], args[1], args[2], args[3]);
  return info.Env().Undefined();
}

// GL_EXPORT void glDrawRangeElements (GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum
// type, const void *indices);
Napi::Value WebGL2RenderingContext::DrawRangeElements(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glDrawRangeElements(args[0], args[1], args[2], args[3], args[4], args[5]);
  return info.Env().Undefined();
}

// GL_EXPORT void glSampleCoverage (GLclampf value, GLboolean invert);
Napi::Value WebGL2RenderingContext::SampleCoverage(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glSampleCoverage(args[0], args[1]);
  return info.Env().Undefined();
}

// GL_EXPORT GLint glGetFragDataLocation (GLuint program, const GLchar* name);
Napi::Value WebGL2RenderingContext::GetFragDataLocation(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  std::string name  = args[1];
  GL_EXPORT::glGetFragDataLocation(args[0], name.data());
  return info.Env().Undefined();
}

Napi::Value WebGL2RenderingContext::GetContextAttributes(Napi::CallbackInfo const& info) {
  return this->context_attributes_.Value();
}

Napi::Value WebGL2RenderingContext::GetClearMask_(Napi::CallbackInfo const& info) {
  return CPPToNapi(info.Env())(this->clear_mask_);
}

void WebGL2RenderingContext::SetClearMask_(Napi::CallbackInfo const& info,
                                           Napi::Value const& value) {
  this->clear_mask_ = NapiToCPP(value);
}

}  // namespace nv
