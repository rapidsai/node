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
#include "napi.h"

#include <algorithm>
#include <map>
#include <nv_node/utilities/args.hpp>
#include <nv_node/utilities/cpp_to_napi.hpp>

namespace nv {

// GLFWAPI GLFWmonitor** glfwGetMonitors(int* count);
Napi::Value glfwGetMonitors(Napi::CallbackInfo const& info) {
  int32_t count{};
  GLFWAPI::GLFWmonitor** mons = GLFWAPI::glfwGetMonitors(&count);
  return CPPToNapi(info)(std::vector<GLFWAPI::GLFWmonitor*>{mons, mons + count});
}

// GLFWAPI GLFWmonitor* glfwGetPrimaryMonitor(void);
Napi::Value glfwGetPrimaryMonitor(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(GLFWAPI::glfwGetPrimaryMonitor());
}

// GLFWAPI void glfwGetMonitorPos(GLFWmonitor* monitor, int* xpos, int* ypos);
Napi::Value glfwGetMonitorPos(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  int32_t x{}, y{};
  GLFW_TRY(env, GLFWAPI::glfwGetMonitorPos(args[0], &x, &y));
  return CPPToNapi(info)(std::map<std::string, int32_t>{//
                                                        {"x", x},
                                                        {"y", y}});
}

// GLFWAPI void glfwGetMonitorWorkarea(GLFWmonitor* monitor, int* xpos, int* ypos, int* width, int*
// height);
Napi::Value glfwGetMonitorWorkarea(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  int32_t x{}, y{}, width{}, height{};
  GLFW_TRY(env, GLFWAPI::glfwGetMonitorWorkarea(args[0], &x, &y, &width, &height));
  return CPPToNapi(info)(std::map<std::string, int32_t>{//
                                                        {"x", x},
                                                        {"y", y},
                                                        {"width", width},
                                                        {"height", height}});
}

// GLFWAPI void glfwGetMonitorPhysicalSize(GLFWmonitor* monitor, int* widthMM, int* heightMM);
Napi::Value glfwGetMonitorPhysicalSize(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  int32_t width{}, height{};
  GLFW_TRY(env, GLFWAPI::glfwGetMonitorPhysicalSize(args[0], &width, &height));
  return CPPToNapi(info)(std::map<std::string, int32_t>{//
                                                        {"width", width},
                                                        {"height", height}});
}

// GLFWAPI void glfwGetMonitorContentScale(GLFWmonitor* monitor, float* xscale, float* yscale);
Napi::Value glfwGetMonitorContentScale(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  float xscale{}, yscale{};
  GLFW_TRY(env, GLFWAPI::glfwGetMonitorContentScale(args[0], &xscale, &yscale));
  return CPPToNapi(info)(std::map<std::string, int32_t>{//
                                                        {"xscale", xscale},
                                                        {"yscale", yscale}});
}

// GLFWAPI const char* glfwGetMonitorName(GLFWmonitor* monitor);
Napi::Value glfwGetMonitorName(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  auto name = GLFWAPI::glfwGetMonitorName(args[0]);
  return CPPToNapi(info)(std::string{name == nullptr ? name : ""});
}

// GLFWAPI void glfwSetMonitorUserPointer(GLFWmonitor* monitor, void* pointer);
// Napi::Value glfwSetMonitorUserPointer(Napi::CallbackInfo const& info);

// GLFWAPI void* glfwGetMonitorUserPointer(GLFWmonitor* monitor);
// Napi::Value glfwGetMonitorUserPointer(Napi::CallbackInfo const& info);

// GLFWAPI const GLFWvidmode* glfwGetVideoModes(GLFWmonitor* monitor, int* count);
Napi::Value glfwGetVideoModes(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  int32_t count{};
  auto modes = GLFWAPI::glfwGetVideoModes(args[0], &count);
  std::vector<Napi::Object> modes_vec(count);
  int32_t i{0};
  std::generate_n(modes_vec.begin(), count, [&]() mutable {
    GLFWvidmode mode = modes[i++];
    return CPPToNapi(info)(std::map<std::string, int>{
      {"width", mode.width},
      {"height", mode.height},
      {"redBits", mode.redBits},
      {"greenBits", mode.greenBits},
      {"blueBits", mode.blueBits},
      {"refreshRate", mode.refreshRate},
    });
  });
  return CPPToNapi(info)(modes_vec);
}

// GLFWAPI const GLFWvidmode* glfwGetVideoMode(GLFWmonitor* monitor);
Napi::Value glfwGetVideoMode(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  auto mode = GLFWAPI::glfwGetVideoMode(args[0]);
  return CPPToNapi(info)(std::map<std::string, int>{
    {"width", mode->width},
    {"height", mode->height},
    {"redBits", mode->redBits},
    {"greenBits", mode->greenBits},
    {"blueBits", mode->blueBits},
    {"refreshRate", mode->refreshRate},
  });
}

// GLFWAPI void glfwSetGamma(GLFWmonitor* monitor, float gamma);
Napi::Value glfwSetGamma(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  GLFW_TRY(env, GLFWAPI::glfwSetGamma(args[0], args[1]));
  return env.Undefined();
}

// GLFWAPI const GLFWgammaramp* glfwGetGammaRamp(GLFWmonitor* monitor);
Napi::Value glfwGetGammaRamp(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  auto ramp    = GLFWAPI::glfwGetGammaRamp(args[0]);
  size_t size  = ramp->size;
  auto js_ramp = Napi::Object::New(env);
  js_ramp.Set("size", CPPToNapi(info)(ramp->size));
  js_ramp.Set("red", CPPToNapi(info)(std::make_tuple(ramp->red, size)));
  js_ramp.Set("green", CPPToNapi(info)(std::make_tuple(ramp->green, size)));
  js_ramp.Set("blue", CPPToNapi(info)(std::make_tuple(ramp->blue, size)));
  return js_ramp;
}

// GLFWAPI void glfwSetGammaRamp(GLFWmonitor* monitor, const GLFWgammaramp* ramp);
Napi::Value glfwSetGammaRamp(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  GLFWmonitor* monitor  = args[0];
  Napi::Object js_ramp  = args[1];
  unsigned int size     = NapiToCPP(js_ramp.Get("size"));
  unsigned short* red   = NapiToCPP(js_ramp.Get("red"));
  unsigned short* green = NapiToCPP(js_ramp.Get("green"));
  unsigned short* blue  = NapiToCPP(js_ramp.Get("blue"));

  GLFWgammaramp ramp{red, green, blue, size};
  GLFW_TRY(env, GLFWAPI::glfwSetGammaRamp(monitor, &ramp));
  return env.Undefined();
}

}  // namespace nv
