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

#include <node_webgl/casting.hpp>
#include <node_webgl/context.hpp>
#include <node_webgl/macros.hpp>

namespace node_webgl {

// GL_EXPORT GLenum glClientWaitSync (GLsync GLsync, GLbitfield flags,GLuint64 timeout);
Napi::Value WebGL2RenderingContext::ClientWaitSync(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glClientWaitSync(FromJS(info[0]), FromJS(info[1]), FromJS(info[2]));
  return env.Undefined();
}

// GL_EXPORT void glDeleteSync (GLsync GLsync);
Napi::Value WebGL2RenderingContext::DeleteSync(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glDeleteSync(FromJS(info[0]));
  return env.Undefined();
}

// GL_EXPORT GLsync glFenceSync (GLenum condition, GLbitfield flags);
Napi::Value WebGL2RenderingContext::FenceSync(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glFenceSync(FromJS(info[0]), FromJS(info[1]));
  return env.Undefined();
}

// GL_EXPORT void glGetSynciv (GLsync GLsync, GLenum pname, GLsizei bufSize, GLsizei* length, GLint
// *values);
Napi::Value WebGL2RenderingContext::GetSyncParameter(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GLint params{};
  GL_EXPORT::glGetSynciv(FromJS(info[0]), FromJS(info[1]), 1, nullptr, &params);
  return ToNapi(env)(params);
}

// GL_EXPORT GLboolean glIsSync (GLsync GLsync);
Napi::Value WebGL2RenderingContext::IsSync(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glIsSync(FromJS(info[0]));
  return env.Undefined();
}

// GL_EXPORT void glWaitSync (GLsync GLsync, GLbitfield flags, GLuint64 timeout);
Napi::Value WebGL2RenderingContext::WaitSync(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glWaitSync(FromJS(info[0]), FromJS(info[1]), FromJS(info[2]));
  return env.Undefined();
}

}  // namespace node_webgl
