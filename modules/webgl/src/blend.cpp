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

// GL_EXPORT void glBlendColor (GLclampf red, GLclampf green, GLclampf blue, GLclampf alpha);
void WebGL2RenderingContext::BlendColor(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glBlendColor(args[0], args[1], args[2], args[3]);
}

// GL_EXPORT void glBlendEquation (GLenum mode);
void WebGL2RenderingContext::BlendEquation(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glBlendEquation(args[0]);
}

// GL_EXPORT void glBlendEquationSeparate (GLenum modeRGB, GLenum modeAlpha);
void WebGL2RenderingContext::BlendEquationSeparate(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glBlendEquationSeparate(args[0], args[1]);
}

// GL_EXPORT void glBlendFunc (GLenum sfactor, GLenum dfactor);
void WebGL2RenderingContext::BlendFunc(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glBlendFunc(args[0], args[1]);
}

// GL_EXPORT void glBlendFuncSeparate (GLenum sfactorRGB, GLenum dfactorRGB, GLenum sfactorAlpha,
// GLenum dfactorAlpha);
void WebGL2RenderingContext::BlendFuncSeparate(Napi::CallbackInfo const& info) {
  CallbackArgs args = info;
  GL_EXPORT::glBlendFuncSeparate(args[0], args[1], args[2], args[3]);
}

}  // namespace nv
