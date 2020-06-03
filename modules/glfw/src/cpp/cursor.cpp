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

#include <node_glfw/casting.hpp>
#include <node_glfw/glfw.hpp>
#include <node_glfw/macros.hpp>

namespace node_glfw {

// GLFWAPI int glfwRawMouseMotionSupported(void);
Napi::Value glfwRawMouseMotionSupported(Napi::CallbackInfo const& info) {
  return ToNapi(info.Env())(static_cast<bool>(GLFWAPI::glfwRawMouseMotionSupported()));
}

// GLFWAPI GLFWcursor* glfwCreateCursor(const GLFWimage* image, int xhot, int yhot);
Napi::Value glfwCreateCursor(Napi::CallbackInfo const& info) {
  auto env                           = info.Env();
  GLFWimage image                    = FromJS(info[0]);
  std::map<std::string, int32_t> pos = FromJS(info[1]);
  return ToNapi(env)(GLFWAPI::glfwCreateCursor(&image, pos["x"], pos["y"]));
}

// GLFWAPI GLFWcursor* glfwCreateStandardCursor(int shape);
Napi::Value glfwCreateStandardCursor(Napi::CallbackInfo const& info) {
  return ToNapi(info.Env())(GLFWAPI::glfwCreateStandardCursor(FromJS(info[0])));
}

// GLFWAPI void glfwDestroyCursor(GLFWcursor* cursor);
Napi::Value glfwDestroyCursor(Napi::CallbackInfo const& info) {
  auto env           = info.Env();
  GLFWcursor* cursor = FromJS(info[0]);
  GLFW_TRY(env, GLFWAPI::glfwDestroyCursor(cursor));
  return env.Undefined();
}

}  // namespace node_glfw
