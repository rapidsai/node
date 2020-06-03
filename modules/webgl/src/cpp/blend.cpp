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

// GL_EXPORT void glBlendColor (GLclampf red, GLclampf green, GLclampf blue, GLclampf alpha);
Napi::Value WebGL2RenderingContext::BlendColor(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glBlendColor(FromJS(info[0]), FromJS(info[1]), FromJS(info[2]), FromJS(info[3]));
  return env.Undefined();
}

// GL_EXPORT void glBlendEquation (GLenum mode);
Napi::Value WebGL2RenderingContext::BlendEquation(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glBlendEquation(FromJS(info[0]));
  return env.Undefined();
}

// GL_EXPORT void glBlendEquationSeparate (GLenum modeRGB, GLenum modeAlpha);
Napi::Value WebGL2RenderingContext::BlendEquationSeparate(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glBlendEquationSeparate(FromJS(info[0]), FromJS(info[1]));
  return env.Undefined();
}

// GL_EXPORT void glBlendFunc (GLenum sfactor, GLenum dfactor);
Napi::Value WebGL2RenderingContext::BlendFunc(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glBlendFunc(FromJS(info[0]), FromJS(info[1]));
  return env.Undefined();
}

// GL_EXPORT void glBlendFuncSeparate (GLenum sfactorRGB, GLenum dfactorRGB, GLenum sfactorAlpha,
// GLenum dfactorAlpha);
Napi::Value WebGL2RenderingContext::BlendFuncSeparate(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GL_EXPORT::glBlendFuncSeparate(
    FromJS(info[0]), FromJS(info[1]), FromJS(info[2]), FromJS(info[3]));
  return env.Undefined();
}

}  // namespace node_webgl
