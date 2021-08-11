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

#include "glfw.hpp"
#include "macros.hpp"

#include <nv_node/utilities/args.hpp>
#include <nv_node/utilities/cpp_to_napi.hpp>

#include <algorithm>

namespace nv {

// GLFWAPI void glfwDefaultWindowHints(void);
void glfwDefaultWindowHints(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GLFW_TRY(env, GLFWAPI::glfwDefaultWindowHints());
}

// GLFWAPI void glfwWindowHint(int hint, int value);
void glfwWindowHint(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  int32_t hint  = args[0];
  int32_t value = args[1];
  GLFW_TRY(env, GLFWAPI::glfwWindowHint(hint, value));
}

// GLFWAPI void glfwWindowHintString(int hint, const char* value);
void glfwWindowHintString(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  int32_t hint      = args[0];
  std::string value = args[1];
  GLFW_TRY(env, GLFWAPI::glfwWindowHintString(hint, value.data()));
}

// GLFWAPI GLFWwindow* glfwCreateWindow(int width, int height, const char* title, GLFWmonitor*
// monitor, GLFWwindow* share);
Napi::Value glfwCreateWindow(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  int32_t width        = args[0];
  int32_t height       = args[1];
  std::string title    = args[2];
  GLFWmonitor* monitor = nullptr;
  GLFWwindow* root_win = nullptr;
  if (info.Length() > 3) { monitor = args[3]; }
  if (info.Length() > 4) { root_win = args[4]; }
  GLFWwindow* window = nullptr;
  GLFW_TRY(env,
           window = GLFWAPI::glfwCreateWindow(width, height, title.c_str(), monitor, root_win));
  return CPPToNapi(info)(window);
}

// #ifdef __linux__

// // void glfwReparentWindow(GLFWwindow* window);
// void glfwReparentWindow(Napi::CallbackInfo const& info) {
//   CallbackArgs args{info};
//   int32_t targetX = args[2];
//   int32_t targetY = args[3];
//   auto display    = glfwGetX11Display();
//   auto child      = glfwGetX11Window(args[0]);
//   Window parent   = args[1];

//   XWindowAttributes attrs;
//   XGetWindowAttributes(display, parent, &attrs);
//   XReparentWindow(display, child, parent, targetX, targetY);
//   XResizeWindow(display, child, attrs.width - targetX, attrs.height - targetY);
// }
// #endif

// GLFWAPI void glfwDestroyWindow(GLFWwindow* window);
void glfwDestroyWindow(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  GLFW_TRY(env, GLFWAPI::glfwDestroyWindow(args[0]));
}

// GLFWAPI int glfwWindowShouldClose(GLFWwindow* window);
Napi::Value glfwWindowShouldClose(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  auto should_close = GLFWAPI::glfwWindowShouldClose(args[0]);
  return CPPToNapi(info)(static_cast<bool>(should_close));
}

// GLFWAPI void glfwSetWindowShouldClose(GLFWwindow* window, int value);
void glfwSetWindowShouldClose(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  GLFW_TRY(env, GLFWAPI::glfwSetWindowShouldClose(args[0], args[1]));
}

// GLFWAPI void glfwSetWindowTitle(GLFWwindow* window, const char* title);
void glfwSetWindowTitle(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  std::string title = args[1];
  GLFW_TRY(env, GLFWAPI::glfwSetWindowTitle(args[0], title.data()));
}

// GLFWAPI void glfwSetWindowIcon(GLFWwindow* window, int count, const GLFWimage* images);
void glfwSetWindowIcon(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  GLFWwindow* window    = args[0];
  Napi::Array js_images = args[1];
  std::vector<GLFWimage> images(js_images.Length());
  size_t i{0};
  std::generate_n(images.begin(), images.size(), [&]() mutable {
    Napi::Object image = js_images.Get(i++).As<Napi::Object>();
    return GLFWimage{
      NapiToCPP(image.Get("width")),
      NapiToCPP(image.Get("height")),
      NapiToCPP(image.Get("pixels")),
    };
  });
  GLFW_TRY(env, GLFWAPI::glfwSetWindowIcon(window, images.size(), images.data()));
}

// GLFWAPI void glfwGetWindowPos(GLFWwindow* window, int* xpos, int* ypos);
Napi::Value glfwGetWindowPos(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  int32_t x{}, y{};
  GLFWwindow* window = args[0];
  GLFW_TRY(env, GLFWAPI::glfwGetWindowPos(window, &x, &y));
  return CPPToNapi(info)(std::map<std::string, int32_t>{{"x", x}, {"y", y}});
}

// GLFWAPI void glfwSetWindowPos(GLFWwindow* window, int xpos, int ypos);
void glfwSetWindowPos(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  GLFWwindow* window                 = args[0];
  std::map<std::string, int32_t> pos = args[1];
  GLFW_TRY(env, GLFWAPI::glfwSetWindowPos(window, pos["x"], pos["y"]));
}

// GLFWAPI void glfwGetWindowSize(GLFWwindow* window, int* width, int* height);
Napi::Value glfwGetWindowSize(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  int32_t width{}, height{};
  GLFWwindow* window = args[0];
  GLFW_TRY(env, GLFWAPI::glfwGetWindowSize(window, &width, &height));
  return CPPToNapi(info)(std::map<std::string, int32_t>{{"width", width}, {"height", height}});
}

// GLFWAPI void glfwSetWindowSizeLimits(GLFWwindow* window, int minwidth, int minheight, int
// maxwidth, int maxheight);
void glfwSetWindowSizeLimits(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  GLFWwindow* window                 = args[0];
  std::map<std::string, int32_t> pos = args[1];
  GLFW_TRY(env,
           GLFWAPI::glfwSetWindowSizeLimits(  //
             window,
             pos["minWidth"],
             pos["minHeight"],
             pos["maxWidth"],
             pos["maxHeight"]));
}

// GLFWAPI void glfwSetWindowAspectRatio(GLFWwindow* window, int numer, int denom);
void glfwSetWindowAspectRatio(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  GLFW_TRY(env, GLFWAPI::glfwSetWindowAspectRatio(args[0], args[1], args[2]));
}

// GLFWAPI void glfwSetWindowSize(GLFWwindow* window, int width, int height);
void glfwSetWindowSize(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  GLFWwindow* window                 = args[0];
  std::map<std::string, int32_t> pos = args[1];
  GLFW_TRY(env, GLFWAPI::glfwSetWindowSize(window, pos["width"], pos["height"]));
}

// GLFWAPI void glfwGetFramebufferSize(GLFWwindow* window, int* width, int* height);
Napi::Value glfwGetFramebufferSize(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  int32_t width{}, height{};
  GLFWwindow* window = args[0];
  GLFW_TRY(env, GLFWAPI::glfwGetFramebufferSize(window, &width, &height));
  return CPPToNapi(info)(std::map<std::string, int32_t>{{"width", width}, {"height", height}});
}

// GLFWAPI void glfwGetWindowFrameSize(GLFWwindow* window, int* left, int* top, int* right, int*
// bottom);
Napi::Value glfwGetWindowFrameSize(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  int32_t left{}, top{}, right{}, bottom{};
  GLFWwindow* window = args[0];
  GLFW_TRY(env, GLFWAPI::glfwGetWindowFrameSize(window, &left, &top, &right, &bottom));
  return CPPToNapi(info)(std::map<std::string, int32_t>{//
                                                        {"left", left},
                                                        {"top", top},
                                                        {"right", right},
                                                        {"bottom", bottom}});
}

// GLFWAPI void glfwGetWindowContentScale(GLFWwindow* window, float* xscale, float* yscale);
Napi::Value glfwGetWindowContentScale(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  float xscale{}, yscale{};
  GLFWwindow* window = args[0];
  GLFW_TRY(env, GLFWAPI::glfwGetWindowContentScale(window, &xscale, &yscale));
  return CPPToNapi(info)(std::map<std::string, int32_t>{//
                                                        {"xscale", xscale},
                                                        {"yscale", yscale}});
}

// GLFWAPI float glfwGetWindowOpacity(GLFWwindow* window);
Napi::Value glfwGetWindowOpacity(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  return CPPToNapi(info)(GLFWAPI::glfwGetWindowOpacity(args[0]));
}

// GLFWAPI void glfwSetWindowOpacity(GLFWwindow* window, float opacity);
void glfwSetWindowOpacity(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  GLFW_TRY(env, GLFWAPI::glfwSetWindowOpacity(args[0], args[1]));
}

// GLFWAPI void glfwIconifyWindow(GLFWwindow* window);
void glfwIconifyWindow(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  GLFW_TRY(env, GLFWAPI::glfwIconifyWindow(args[0]));
}

// GLFWAPI void glfwRestoreWindow(GLFWwindow* window);
void glfwRestoreWindow(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  GLFW_TRY(env, GLFWAPI::glfwRestoreWindow(args[0]));
}

// GLFWAPI void glfwMaximizeWindow(GLFWwindow* window);
void glfwMaximizeWindow(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  GLFW_TRY(env, GLFWAPI::glfwMaximizeWindow(args[0]));
}

// GLFWAPI void glfwShowWindow(GLFWwindow* window);
void glfwShowWindow(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  GLFW_TRY(env, GLFWAPI::glfwShowWindow(args[0]));
}

// GLFWAPI void glfwHideWindow(GLFWwindow* window);
void glfwHideWindow(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  GLFW_TRY(env, GLFWAPI::glfwHideWindow(args[0]));
}

// GLFWAPI void glfwFocusWindow(GLFWwindow* window);
void glfwFocusWindow(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  GLFW_TRY(env, GLFWAPI::glfwFocusWindow(args[0]));
}

// GLFWAPI void glfwRequestWindowAttention(GLFWwindow* window);
void glfwRequestWindowAttention(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  GLFW_TRY(env, GLFWAPI::glfwRequestWindowAttention(args[0]));
}

// GLFWAPI GLFWmonitor* glfwGetWindowMonitor(GLFWwindow* window);
Napi::Value glfwGetWindowMonitor(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  return CPPToNapi(info)(GLFWAPI::glfwGetWindowMonitor(args[0]));
}

// GLFWAPI void glfwSetWindowMonitor(GLFWwindow* window, GLFWmonitor* monitor, int xpos, int ypos,
// int width, int height, int refreshRate);
void glfwSetWindowMonitor(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  GLFWwindow* window                 = args[0];
  GLFWmonitor* monitor               = args[1];
  std::map<std::string, int32_t> map = args[2];
  GLFW_TRY(env,
           GLFWAPI::glfwSetWindowMonitor(  //
             window,
             monitor,
             map["x"],
             map["y"],
             map["width"],
             map["height"],
             map["refreshRate"]));
}

// GLFWAPI int glfwGetWindowAttrib(GLFWwindow* window, int attrib);
Napi::Value glfwGetWindowAttrib(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  return CPPToNapi(info)(GLFWAPI::glfwGetWindowAttrib(args[0], args[1]));
}

// GLFWAPI void glfwSetWindowAttrib(GLFWwindow* window, int attrib, int value);
void glfwSetWindowAttrib(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  GLFWwindow* window = args[0];
  int32_t attrib     = args[1];
  int32_t value      = args[2];
  GLFW_TRY(env, GLFWAPI::glfwSetWindowAttrib(window, attrib, value));
}

// GLFWAPI void glfwSetWindowUserPointer(GLFWwindow* window, void* pointer);
// Napi::Value glfwSetWindowUserPointer(Napi::CallbackInfo const& info);

// GLFWAPI void* glfwGetWindowUserPointer(GLFWwindow* window);
// Napi::Value glfwGetWindowUserPointer(Napi::CallbackInfo const& info);

// GLFWAPI int glfwGetInputMode(GLFWwindow* window, int mode);
Napi::Value glfwGetInputMode(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  return CPPToNapi(info)(GLFWAPI::glfwGetInputMode(args[0], args[1]));
}

// GLFWAPI void glfwSetInputMode(GLFWwindow* window, int mode, int value);
void glfwSetInputMode(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  GLFW_TRY(env, GLFWAPI::glfwSetInputMode(args[0], args[1], args[2]));
}

// GLFWAPI int glfwGetKey(GLFWwindow* window, int key);
Napi::Value glfwGetKey(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  return CPPToNapi(info)(GLFWAPI::glfwGetKey(args[0], args[1]));
}

// GLFWAPI int glfwGetMouseButton(GLFWwindow* window, int button);
Napi::Value glfwGetMouseButton(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  return CPPToNapi(info)(GLFWAPI::glfwGetMouseButton(args[0], args[1]));
}

// GLFWAPI void glfwGetCursorPos(GLFWwindow* window, double* xpos, double* ypos);
Napi::Value glfwGetCursorPos(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  double x{}, y{};
  GLFWwindow* window = args[0];
  GLFW_TRY(env, GLFWAPI::glfwGetCursorPos(window, &x, &y));
  return CPPToNapi(info)(std::map<std::string, int32_t>{//
                                                        {"x", x},
                                                        {"y", y}});
}

// GLFWAPI void glfwSetCursorPos(GLFWwindow* window, double xpos, double ypos);
void glfwSetCursorPos(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  GLFWwindow* window                 = args[0];
  std::map<std::string, int32_t> pos = args[1];
  GLFW_TRY(env, GLFWAPI::glfwSetCursorPos(window, pos["x"], pos["y"]));
}

// GLFWAPI void glfwSetCursor(GLFWwindow* window, GLFWcursor* cursor);
void glfwSetCursor(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  GLFW_TRY(env, GLFWAPI::glfwSetCursor(args[0], args[1]));
}

// GLFWAPI void glfwSetClipboardString(GLFWwindow* window, const char* string);
void glfwSetClipboardString(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  std::string str = args[1];
  GLFW_TRY(env, GLFWAPI::glfwSetClipboardString(args[0], str.data()));
}

// GLFWAPI const char* glfwGetClipboardString(GLFWwindow* window);
Napi::Value glfwGetClipboardString(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  auto str = GLFWAPI::glfwGetClipboardString(args[0]);
  return CPPToNapi(info)(std::string{str == nullptr ? str : ""});
}

// GLFWAPI void glfwMakeContextCurrent(GLFWwindow* window);
void glfwMakeContextCurrent(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  GLFW_TRY(env, GLFWAPI::glfwMakeContextCurrent(args[0]));
}

// GLFWAPI GLFWwindow* glfwGetCurrentContext(void);
Napi::Value glfwGetCurrentContext(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(GLFWAPI::glfwGetCurrentContext());
}

// GLFWAPI void glfwSwapBuffers(GLFWwindow* window);
void glfwSwapBuffers(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  GLFW_TRY(env, GLFWAPI::glfwSwapBuffers(args[0]));
}

}  // namespace nv
