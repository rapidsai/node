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

#include "glfw.hpp"
#include "macros.hpp"

#include <nv_node/utilities/args.hpp>
#include <nv_node/utilities/cpp_to_napi.hpp>

namespace nv {

// GLFWAPI int glfwRawMouseMotionSupported(void);
Napi::Value glfwRawMouseMotionSupported(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(static_cast<bool>(GLFWAPI::glfwRawMouseMotionSupported()));
}

// GLFWAPI GLFWcursor* glfwCreateCursor(const GLFWimage* image, int xhot, int yhot);
Napi::Value glfwCreateCursor(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  auto obj = info[0].As<Napi::Object>();
  GLFWimage image{//
                  NapiToCPP(obj.Get("width")),
                  NapiToCPP(obj.Get("height")),
                  NapiToCPP(obj.Get("pixels"))};
  std::map<std::string, int32_t> pos = args[1];
  return CPPToNapi(info)(GLFWAPI::glfwCreateCursor(&image, pos["x"], pos["y"]));
}

// GLFWAPI GLFWcursor* glfwCreateStandardCursor(int shape);
Napi::Value glfwCreateStandardCursor(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  GLFWcursor* cursor = GLFWAPI::glfwCreateStandardCursor(args[0]);
  return CPPToNapi(info)(cursor);
}

// GLFWAPI void glfwDestroyCursor(GLFWcursor* cursor);
Napi::Value glfwDestroyCursor(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  GLFWcursor* cursor = args[0];
  GLFW_TRY(env, GLFWAPI::glfwDestroyCursor(cursor));
  return env.Undefined();
}

}  // namespace nv
