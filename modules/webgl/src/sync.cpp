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

// GL_EXPORT GLenum glClientWaitSync (GLsync GLsync, GLbitfield flags,GLuint64 timeout);
Napi::Value WebGL2RenderingContext::ClientWaitSync(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  return CPPToNapi(info)(
    GL_EXPORT::glClientWaitSync(*WebGLSync::Unwrap(args[0].ToObject()), args[1], args[2]));
}

// GL_EXPORT void glDeleteSync (GLsync GLsync);
Napi::Value WebGL2RenderingContext::DeleteSync(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glDeleteSync(args[0]);
  return info.Env().Undefined();
}

// GL_EXPORT GLsync glFenceSync (GLenum condition, GLbitfield flags);
Napi::Value WebGL2RenderingContext::FenceSync(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  return WebGLSync::New(GL_EXPORT::glFenceSync(args[0], args[1]));
}

// GL_EXPORT void glGetSynciv (GLsync GLsync, GLenum pname, GLsizei bufSize, GLsizei* length, GLint
// *values);
Napi::Value WebGL2RenderingContext::GetSyncParameter(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GLint params{};
  GL_EXPORT::glGetSynciv(args[0], args[1], 1, nullptr, &params);
  return CPPToNapi(info)(params);
}

// GL_EXPORT GLboolean glIsSync (GLsync GLsync);
Napi::Value WebGL2RenderingContext::IsSync(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  return CPPToNapi(info)(GL_EXPORT::glIsSync(args[0]));
}

// GL_EXPORT void glWaitSync (GLsync GLsync, GLbitfield flags, GLuint64 timeout);
Napi::Value WebGL2RenderingContext::WaitSync(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glWaitSync(args[0], args[1], args[2]);
  return info.Env().Undefined();
}

}  // namespace nv
