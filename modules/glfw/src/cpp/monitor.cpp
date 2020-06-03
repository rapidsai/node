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

// GLFWAPI GLFWmonitor** glfwGetMonitors(int* count);
Napi::Value glfwGetMonitors(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  int32_t count{};
  GLFWAPI::GLFWmonitor** mons = GLFWAPI::glfwGetMonitors(&count);
  return ToNapi(env)(std::vector<GLFWAPI::GLFWmonitor*>{mons, mons + count});
}

// GLFWAPI GLFWmonitor* glfwGetPrimaryMonitor(void);
Napi::Value glfwGetPrimaryMonitor(Napi::CallbackInfo const& info) {
  return ToNapi(info.Env())(GLFWAPI::glfwGetPrimaryMonitor());
}

// GLFWAPI void glfwGetMonitorPos(GLFWmonitor* monitor, int* xpos, int* ypos);
Napi::Value glfwGetMonitorPos(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  int32_t x{}, y{};
  GLFW_TRY(env, GLFWAPI::glfwGetMonitorPos(FromJS(info[0]), &x, &y));
  return ToNapi(env)(std::map<std::string, int32_t>{{"x", x}, {"y", y}});
}

// GLFWAPI void glfwGetMonitorWorkarea(GLFWmonitor* monitor, int* xpos, int* ypos, int* width, int*
// height);
Napi::Value glfwGetMonitorWorkarea(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  int32_t x{}, y{}, width{}, height{};
  GLFW_TRY(env, GLFWAPI::glfwGetMonitorWorkarea(FromJS(info[0]), &x, &y, &width, &height));
  return ToNapi(env)(
    std::map<std::string, int32_t>{{"x", x}, {"y", y}, {"width", width}, {"height", height}});
}

// GLFWAPI void glfwGetMonitorPhysicalSize(GLFWmonitor* monitor, int* widthMM, int* heightMM);
Napi::Value glfwGetMonitorPhysicalSize(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  int32_t width{}, height{};
  GLFW_TRY(env, GLFWAPI::glfwGetMonitorPhysicalSize(FromJS(info[0]), &width, &height));
  return ToNapi(env)(std::map<std::string, int32_t>{{"width", width}, {"height", height}});
}

// GLFWAPI void glfwGetMonitorContentScale(GLFWmonitor* monitor, float* xscale, float* yscale);
Napi::Value glfwGetMonitorContentScale(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  float xscale{}, yscale{};
  GLFW_TRY(env, GLFWAPI::glfwGetMonitorContentScale(FromJS(info[0]), &xscale, &yscale));
  return ToNapi(env)(std::map<std::string, int32_t>{{"xscale", xscale}, {"yscale", yscale}});
}

// GLFWAPI const char* glfwGetMonitorName(GLFWmonitor* monitor);
Napi::Value glfwGetMonitorName(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  return ToNapi(env)(GLFWAPI::glfwGetMonitorName(FromJS(info[0])));
}

// GLFWAPI void glfwSetMonitorUserPointer(GLFWmonitor* monitor, void* pointer);
// Napi::Value glfwSetMonitorUserPointer(Napi::CallbackInfo const& info);

// GLFWAPI void* glfwGetMonitorUserPointer(GLFWmonitor* monitor);
// Napi::Value glfwGetMonitorUserPointer(Napi::CallbackInfo const& info);

// GLFWAPI const GLFWvidmode* glfwGetVideoModes(GLFWmonitor* monitor, int* count);
Napi::Value glfwGetVideoModes(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  int32_t count{};
  auto modes = GLFWAPI::glfwGetVideoModes(FromJS(info[0]), &count);
  return ToNapi(env)(std::vector<GLFWAPI::GLFWvidmode>{modes, modes + count});
}

// GLFWAPI const GLFWvidmode* glfwGetVideoMode(GLFWmonitor* monitor);
Napi::Value glfwGetVideoMode(Napi::CallbackInfo const& info) {
  return ToNapi(info.Env())(GLFWAPI::glfwGetVideoMode(FromJS(info[0])));
}

// GLFWAPI void glfwSetGamma(GLFWmonitor* monitor, float gamma);
Napi::Value glfwSetGamma(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GLFW_TRY(env, GLFWAPI::glfwSetGamma(FromJS(info[0]), FromJS(info[1])));
  return env.Undefined();
}

// GLFWAPI const GLFWgammaramp* glfwGetGammaRamp(GLFWmonitor* monitor);
Napi::Value glfwGetGammaRamp(Napi::CallbackInfo const& info) {
  return ToNapi(info.Env())(GLFWAPI::glfwGetGammaRamp(FromJS(info[0])));
}

// GLFWAPI void glfwSetGammaRamp(GLFWmonitor* monitor, const GLFWgammaramp* ramp);
Napi::Value glfwSetGammaRamp(Napi::CallbackInfo const& info) {
  auto env                 = info.Env();
  GLFWmonitor* monitor     = FromJS(info[0]);
  GLFWgammaramp const ramp = FromJS(info[1]);
  GLFW_TRY(env, GLFWAPI::glfwSetGammaRamp(monitor, &ramp));
  return env.Undefined();
}

}  // namespace node_glfw
