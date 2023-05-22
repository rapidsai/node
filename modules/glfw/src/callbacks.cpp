// Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <nv_node/objectwrap.hpp>

#include <napi.h>

#include <map>
#include <vector>

namespace Napi {

template <>
inline Value Value::From(napi_env env, GLFWmonitor* const& monitor) {
  return Napi::Number::New(env, reinterpret_cast<uintptr_t>(monitor));
}

template <>
inline Value Value::From(napi_env env, GLFWwindow* const& window) {
  return Napi::Number::New(env, reinterpret_cast<uintptr_t>(window));
}

template <>
inline Value Value::From(napi_env env, std::vector<std::string> const& vec) {
  uint32_t idx = 0;
  auto arr     = Napi::Array::New(env, vec.size());
  for (auto const& val : vec) { arr[idx++] = val; }
  return arr;
}

}  // namespace Napi

namespace nv {

namespace {

Napi::FunctionReference error_cb;
Napi::FunctionReference joystick_cb;
Napi::FunctionReference monitor_cb;

std::map<GLFWwindow*, Napi::FunctionReference> pos_cbs;
std::map<GLFWwindow*, Napi::FunctionReference> size_cbs;
std::map<GLFWwindow*, Napi::FunctionReference> close_cbs;
std::map<GLFWwindow*, Napi::FunctionReference> refresh_cbs;
std::map<GLFWwindow*, Napi::FunctionReference> focus_cbs;
std::map<GLFWwindow*, Napi::FunctionReference> iconify_cbs;
std::map<GLFWwindow*, Napi::FunctionReference> maximize_cbs;
std::map<GLFWwindow*, Napi::FunctionReference> scale_cbs;
std::map<GLFWwindow*, Napi::FunctionReference> fb_resize_cbs;
std::map<GLFWwindow*, Napi::FunctionReference> key_cbs;
std::map<GLFWwindow*, Napi::FunctionReference> char_cbs;
std::map<GLFWwindow*, Napi::FunctionReference> charmods_cbs;
std::map<GLFWwindow*, Napi::FunctionReference> mousebutton_cbs;
std::map<GLFWwindow*, Napi::FunctionReference> cursorpos_cbs;
std::map<GLFWwindow*, Napi::FunctionReference> cursorenter_cbs;
std::map<GLFWwindow*, Napi::FunctionReference> scroll_cbs;
std::map<GLFWwindow*, Napi::FunctionReference> drop_cbs;

template <typename... Args>
std::vector<napi_value> to_napi(Napi::Env env, Args&&... args) {
  std::vector<napi_value> napi_values;

  napi_values.reserve(sizeof...(Args));

  detail::tuple::for_each(
    std::make_tuple<Args...>(std::forward<Args>(args)...),
    [&](auto const& x) mutable { napi_values.push_back(Napi::Value::From(env, x)); });

  return napi_values;
}

void GLFWerror_cb(int32_t err, const char* msg) {
  auto& cb = error_cb;
  if (!cb.IsEmpty()) {  //
    auto args = to_napi(cb.Env(), err, std::string{msg == nullptr ? msg : ""});
    cb.MakeCallback(cb.Env().Global(), args);
  }
}

void GLFWmonitor_cb(GLFWmonitor* monitor, int32_t event) {
  auto& cb = monitor_cb;
  if (!cb.IsEmpty()) {  //
    auto args = to_napi(cb.Env(), monitor, event);
    cb.MakeCallback(cb.Env().Global(), args);
  }
}

void GLFWjoystick_cb(int32_t joystick, int32_t event) {
  auto& cb = joystick_cb;
  if (!cb.IsEmpty()) {  //
    auto args = to_napi(cb.Env(), joystick, event);
    cb.MakeCallback(cb.Env().Global(), args);
  }
}

void GLFWwindowpos_cb(GLFWwindow* window, int32_t x, int32_t y) {
  if (pos_cbs.find(window) != pos_cbs.end()) {
    auto& cb = pos_cbs[window];
    if (!cb.IsEmpty()) {  //
      auto args = to_napi(cb.Env(), window, x, y);
      cb.MakeCallback(cb.Env().Global(), args);
    }
  }
}

void GLFWwindowsize_cb(GLFWwindow* window, int32_t width, int32_t height) {
  if (size_cbs.find(window) != size_cbs.end()) {
    auto& cb = size_cbs[window];
    if (!cb.IsEmpty()) {  //
      auto args = to_napi(cb.Env(), window, width, height);
      cb.MakeCallback(cb.Env().Global(), args);
    }
  }
}

void GLFWwindowclose_cb(GLFWwindow* window) {
  if (close_cbs.find(window) != close_cbs.end()) {
    auto& cb = close_cbs[window];
    if (!cb.IsEmpty()) {  //
      auto args = to_napi(cb.Env(), window);
      cb.MakeCallback(cb.Env().Global(), args);
    }
  }
}

void GLFWwindowrefresh_cb(GLFWwindow* window) {
  if (refresh_cbs.find(window) != refresh_cbs.end()) {
    auto& cb = refresh_cbs[window];
    if (!cb.IsEmpty()) {  //
      auto args = to_napi(cb.Env(), window);
      cb.MakeCallback(cb.Env().Global(), args);
    }
  }
}

void GLFWwindowfocus_cb(GLFWwindow* window, int32_t focused) {
  if (focus_cbs.find(window) != focus_cbs.end()) {
    auto& cb = focus_cbs[window];
    if (!cb.IsEmpty()) {  //
      auto args = to_napi(cb.Env(), window, focused);
      cb.MakeCallback(cb.Env().Global(), args);
    }
  }
}

void GLFWwindowiconify_cb(GLFWwindow* window, int32_t iconified) {
  if (iconify_cbs.find(window) != iconify_cbs.end()) {
    auto& cb = iconify_cbs[window];
    if (!cb.IsEmpty()) {  //
      auto args = to_napi(cb.Env(), window, iconified);
      cb.MakeCallback(cb.Env().Global(), args);
    }
  }
}

void GLFWwindowmaximize_cb(GLFWwindow* window, int32_t maximized) {
  if (maximize_cbs.find(window) != maximize_cbs.end()) {
    auto& cb = maximize_cbs[window];
    if (!cb.IsEmpty()) {  //
      auto args = to_napi(cb.Env(), window, maximized);
      cb.MakeCallback(cb.Env().Global(), args);
    }
  }
}

void GLFWframebuffersize_cb(GLFWwindow* window, int32_t width, int32_t height) {
  if (fb_resize_cbs.find(window) != fb_resize_cbs.end()) {
    auto& cb  = fb_resize_cbs[window];
    auto args = to_napi(cb.Env(), window, width, height);
    cb.MakeCallback(cb.Env().Global(), args);
  }
}

void GLFWwindowcontentscale_cb(GLFWwindow* window, float xscale, float yscale) {
  if (scale_cbs.find(window) != scale_cbs.end()) {
    auto& cb = scale_cbs[window];
    if (!cb.IsEmpty()) {  //
      auto args = to_napi(cb.Env(), window, xscale, yscale);
      cb.MakeCallback(cb.Env().Global(), args);
    }
  }
}

void GLFWkey_cb(GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods) {
  if (key_cbs.find(window) != key_cbs.end()) {
    auto& cb = key_cbs[window];
    if (!cb.IsEmpty()) {  //
      auto args = to_napi(cb.Env(), window, key, scancode, action, mods);
      cb.MakeCallback(cb.Env().Global(), args);
    }
  }
}

void GLFWchar_cb(GLFWwindow* window, uint32_t codepoint) {
  if (char_cbs.find(window) != char_cbs.end()) {
    auto& cb = char_cbs[window];
    if (!cb.IsEmpty()) {  //
      auto args = to_napi(cb.Env(), window, codepoint);
      cb.MakeCallback(cb.Env().Global(), args);
    }
  }
}

void GLFWcharmods_cb(GLFWwindow* window, uint32_t codepoint, int32_t mods) {
  if (charmods_cbs.find(window) != charmods_cbs.end()) {
    auto& cb = charmods_cbs[window];
    if (!cb.IsEmpty()) {  //
      auto args = to_napi(cb.Env(), window, codepoint, mods);
      cb.MakeCallback(cb.Env().Global(), args);
    }
  }
}

void GLFWmousebutton_cb(GLFWwindow* window, int32_t button, int32_t action, int32_t mods) {
  if (mousebutton_cbs.find(window) != mousebutton_cbs.end()) {
    auto& cb = mousebutton_cbs[window];
    if (!cb.IsEmpty()) {  //
      auto args = to_napi(cb.Env(), window, button, action, mods);
      cb.MakeCallback(cb.Env().Global(), args);
    }
  }
}

void GLFWcursorpos_cb(GLFWwindow* window, double x, double y) {
  if (cursorpos_cbs.find(window) != cursorpos_cbs.end()) {
    auto& cb = cursorpos_cbs[window];
    if (!cb.IsEmpty()) {  //
      auto args = to_napi(cb.Env(), window, x, y);
      cb.MakeCallback(cb.Env().Global(), args);
    }
  }
}

void GLFWcursorenter_cb(GLFWwindow* window, int32_t entered) {
  if (cursorenter_cbs.find(window) != cursorenter_cbs.end()) {
    auto& cb = cursorenter_cbs[window];
    if (!cb.IsEmpty()) {  //
      auto args = to_napi(cb.Env(), window, entered);
      cb.MakeCallback(cb.Env().Global(), args);
    }
  }
}

void GLFWscroll_cb(GLFWwindow* window, double x, double y) {
  if (scroll_cbs.find(window) != scroll_cbs.end()) {
    auto& cb = scroll_cbs[window];
    if (!cb.IsEmpty()) {  //
      auto args = to_napi(cb.Env(), window, x, y);
      cb.MakeCallback(cb.Env().Global(), args);
    }
  }
}

void GLFWdrop_cb(GLFWwindow* window, int count, const char** paths) {
  if (drop_cbs.find(window) != drop_cbs.end()) {
    auto& cb = drop_cbs[window];
    if (!cb.IsEmpty()) {  //
      auto args = to_napi(cb.Env(), window, std::vector<std::string>{paths, paths + count});
      cb.MakeCallback(cb.Env().Global(), args);
    }
  }
}

};  // namespace

// GLFWAPI GLFWerrorfun glfwSetErrorCallback(GLFWerrorfun callback);
void glfwSetErrorCallback(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  if (info[0].IsFunction()) {
    error_cb = Napi::Persistent(info[0].As<Napi::Function>());
    GLFW_TRY(env, GLFWAPI::glfwSetErrorCallback(GLFWerror_cb));
  } else {
    error_cb.Reset();
    GLFW_TRY(env, GLFWAPI::glfwSetErrorCallback(NULL));
  }
}

// GLFWAPI GLFWmonitorfun glfwSetMonitorCallback(GLFWmonitorfun callback);
void glfwSetMonitorCallback(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  if (info[0].IsFunction()) {
    monitor_cb = Napi::Persistent(info[0].As<Napi::Function>());
    GLFW_TRY(env, GLFWAPI::glfwSetMonitorCallback(GLFWmonitor_cb));
  } else {
    monitor_cb.Reset();
    GLFW_TRY(env, GLFWAPI::glfwSetMonitorCallback(NULL));
  }
}

// GLFWAPI GLFWjoystickfun glfwSetJoystickCallback(GLFWjoystickfun callback);
void glfwSetJoystickCallback(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  if (info[0].IsFunction()) {
    joystick_cb = Napi::Persistent(info[0].As<Napi::Function>());
    GLFW_TRY(env, GLFWAPI::glfwSetJoystickCallback(GLFWjoystick_cb));
  } else {
    joystick_cb.Reset();
    GLFW_TRY(env, GLFWAPI::glfwSetJoystickCallback(NULL));
  }
}

// GLFWAPI GLFWwindowposfun glfwSetWindowPosCallback(GLFWwindow* window, GLFWwindowposfun callback);
void glfwSetWindowPosCallback(Napi::CallbackInfo const& info) {
  auto window = reinterpret_cast<GLFWwindow*>(info[0].ToNumber().Int64Value());
  if (window != nullptr) {  //
    auto env = info.Env();
    if (info[1].IsFunction()) {
      pos_cbs[window] = Napi::Persistent(info[1].As<Napi::Function>());
      GLFW_TRY(env, GLFWAPI::glfwSetWindowPosCallback(window, GLFWwindowpos_cb));
    } else {
      if (pos_cbs.find(window) != pos_cbs.end()) { pos_cbs.erase(window); }
      GLFW_TRY(env, GLFWAPI::glfwSetWindowPosCallback(window, NULL));
    }
  }
}

// GLFWAPI GLFWwindowsizefun glfwSetWindowSizeCallback(GLFWwindow* window, GLFWwindowsizefun
// callback);
void glfwSetWindowSizeCallback(Napi::CallbackInfo const& info) {
  auto window = reinterpret_cast<GLFWwindow*>(info[0].ToNumber().Int64Value());
  if (window != nullptr) {  //
    auto env = info.Env();
    if (info[1].IsFunction()) {
      size_cbs[window] = Napi::Persistent(info[1].As<Napi::Function>());
      GLFW_TRY(env, GLFWAPI::glfwSetWindowSizeCallback(window, GLFWwindowsize_cb));
    } else {
      if (size_cbs.find(window) != size_cbs.end()) { size_cbs.erase(window); }
      GLFW_TRY(env, GLFWAPI::glfwSetWindowSizeCallback(window, NULL));
    }
  }
}

// GLFWAPI GLFWwindowclosefun glfwSetWindowCloseCallback(GLFWwindow* window, GLFWwindowclosefun
// callback);
void glfwSetWindowCloseCallback(Napi::CallbackInfo const& info) {
  auto window = reinterpret_cast<GLFWwindow*>(info[0].ToNumber().Int64Value());
  if (window != nullptr) {  //
    auto env = info.Env();
    if (info[1].IsFunction()) {
      close_cbs[window] = Napi::Persistent(info[1].As<Napi::Function>());
      GLFW_TRY(env, GLFWAPI::glfwSetWindowCloseCallback(window, GLFWwindowclose_cb));
    } else {
      if (close_cbs.find(window) != close_cbs.end()) { close_cbs.erase(window); }
      GLFW_TRY(env, GLFWAPI::glfwSetWindowCloseCallback(window, NULL));
    }
  }
}

// GLFWAPI GLFWwindowrefreshfun glfwSetWindowRefreshCallback(GLFWwindow* window,
// GLFWwindowrefreshfun callback);
void glfwSetWindowRefreshCallback(Napi::CallbackInfo const& info) {
  auto window = reinterpret_cast<GLFWwindow*>(info[0].ToNumber().Int64Value());
  if (window != nullptr) {  //
    auto env = info.Env();
    if (info[1].IsFunction()) {
      refresh_cbs[window] = Napi::Persistent(info[1].As<Napi::Function>());
      GLFW_TRY(env, GLFWAPI::glfwSetWindowRefreshCallback(window, GLFWwindowrefresh_cb));
    } else {
      if (refresh_cbs.find(window) != refresh_cbs.end()) { refresh_cbs.erase(window); }
      GLFW_TRY(env, GLFWAPI::glfwSetWindowRefreshCallback(window, NULL));
    }
  }
}

// GLFWAPI GLFWwindowfocusfun glfwSetWindowFocusCallback(GLFWwindow* window, GLFWwindowfocusfun
// callback);
void glfwSetWindowFocusCallback(Napi::CallbackInfo const& info) {
  auto window = reinterpret_cast<GLFWwindow*>(info[0].ToNumber().Int64Value());
  if (window != nullptr) {  //
    auto env = info.Env();
    if (info[1].IsFunction()) {
      focus_cbs[window] = Napi::Persistent(info[1].As<Napi::Function>());
      GLFW_TRY(env, GLFWAPI::glfwSetWindowFocusCallback(window, GLFWwindowfocus_cb));
    } else {
      if (focus_cbs.find(window) != focus_cbs.end()) { focus_cbs.erase(window); }
      GLFW_TRY(env, GLFWAPI::glfwSetWindowFocusCallback(window, NULL));
    }
  }
}

// GLFWAPI GLFWwindowiconifyfun glfwSetWindowIconifyCallback(GLFWwindow* window,
// GLFWwindowiconifyfun callback);
void glfwSetWindowIconifyCallback(Napi::CallbackInfo const& info) {
  auto window = reinterpret_cast<GLFWwindow*>(info[0].ToNumber().Int64Value());
  if (window != nullptr) {  //
    auto env = info.Env();
    if (info[1].IsFunction()) {
      iconify_cbs[window] = Napi::Persistent(info[1].As<Napi::Function>());
      GLFW_TRY(env, GLFWAPI::glfwSetWindowIconifyCallback(window, GLFWwindowiconify_cb));
    } else {
      if (iconify_cbs.find(window) != iconify_cbs.end()) { iconify_cbs.erase(window); }
      GLFW_TRY(env, GLFWAPI::glfwSetWindowIconifyCallback(window, NULL));
    }
  }
}

// GLFWAPI GLFWwindowmaximizefun glfwSetWindowMaximizeCallback(GLFWwindow* window,
// GLFWwindowmaximizefun callback);
void glfwSetWindowMaximizeCallback(Napi::CallbackInfo const& info) {
  auto window = reinterpret_cast<GLFWwindow*>(info[0].ToNumber().Int64Value());
  if (window != nullptr) {  //
    auto env = info.Env();
    if (info[1].IsFunction()) {
      maximize_cbs[window] = Napi::Persistent(info[1].As<Napi::Function>());
      GLFW_TRY(env, GLFWAPI::glfwSetWindowMaximizeCallback(window, GLFWwindowmaximize_cb));
    } else {
      if (maximize_cbs.find(window) != maximize_cbs.end()) { maximize_cbs.erase(window); }
      GLFW_TRY(env, GLFWAPI::glfwSetWindowMaximizeCallback(window, NULL));
    }
  }
}

// GLFWAPI GLFWframebuffersizefun glfwSetFramebufferSizeCallback(GLFWwindow* window,
// GLFWframebuffersizefun callback);
void glfwSetFramebufferSizeCallback(Napi::CallbackInfo const& info) {
  auto window = reinterpret_cast<GLFWwindow*>(info[0].ToNumber().Int64Value());
  if (window != nullptr) {  //
    auto env = info.Env();
    if (info[1].IsFunction()) {
      fb_resize_cbs[window] = Napi::Persistent(info[1].As<Napi::Function>());
      GLFW_TRY(env, GLFWAPI::glfwSetFramebufferSizeCallback(window, GLFWframebuffersize_cb));
    } else {
      if (fb_resize_cbs.find(window) != fb_resize_cbs.end()) { fb_resize_cbs.erase(window); }
      GLFW_TRY(env, GLFWAPI::glfwSetFramebufferSizeCallback(window, NULL));
    }
  }
}

// GLFWAPI GLFWwindowcontentscalefun glfwSetWindowContentScaleCallback(GLFWwindow* window,
// GLFWwindowcontentscalefun callback);
void glfwSetWindowContentScaleCallback(Napi::CallbackInfo const& info) {
  auto window = reinterpret_cast<GLFWwindow*>(info[0].ToNumber().Int64Value());
  if (window != nullptr) {  //
    auto env = info.Env();
    if (info[1].IsNull() || info[1].IsEmpty() || info[1].IsUndefined()) {
      if (info[1].IsFunction()) {
        scale_cbs[window] = Napi::Persistent(info[1].As<Napi::Function>());
        GLFW_TRY(env,
                 GLFWAPI::glfwSetWindowContentScaleCallback(window, GLFWwindowcontentscale_cb));
      } else {
        if (scale_cbs.find(window) != scale_cbs.end()) { scale_cbs.erase(window); }
        GLFW_TRY(env, GLFWAPI::glfwSetWindowContentScaleCallback(window, NULL));
      }
    }
  }
}

// GLFWAPI GLFWkeyfun glfwSetKeyCallback(GLFWwindow* window, GLFWkeyfun callback);
void glfwSetKeyCallback(Napi::CallbackInfo const& info) {
  auto window = reinterpret_cast<GLFWwindow*>(info[0].ToNumber().Int64Value());
  if (window != nullptr) {  //
    auto env = info.Env();
    if (info[1].IsFunction()) {
      key_cbs[window] = Napi::Persistent(info[1].As<Napi::Function>());
      GLFW_TRY(env, GLFWAPI::glfwSetKeyCallback(window, GLFWkey_cb));
    } else {
      if (key_cbs.find(window) != key_cbs.end()) { key_cbs.erase(window); }
      GLFW_TRY(env, GLFWAPI::glfwSetKeyCallback(window, NULL));
    }
  }
}

// GLFWAPI GLFWcharfun glfwSetCharCallback(GLFWwindow* window, GLFWcharfun callback);
void glfwSetCharCallback(Napi::CallbackInfo const& info) {
  auto window = reinterpret_cast<GLFWwindow*>(info[0].ToNumber().Int64Value());
  if (window != nullptr) {  //
    auto env = info.Env();
    if (info[1].IsFunction()) {
      char_cbs[window] = Napi::Persistent(info[1].As<Napi::Function>());
      GLFW_TRY(env, GLFWAPI::glfwSetCharCallback(window, GLFWchar_cb));
    } else {
      if (char_cbs.find(window) != char_cbs.end()) { char_cbs.erase(window); }
      GLFW_TRY(env, GLFWAPI::glfwSetCharCallback(window, NULL));
    }
  }
}

// GLFWAPI GLFWcharmodsfun glfwSetCharModsCallback(GLFWwindow* window, GLFWcharmodsfun callback);
void glfwSetCharModsCallback(Napi::CallbackInfo const& info) {
  auto window = reinterpret_cast<GLFWwindow*>(info[0].ToNumber().Int64Value());
  if (window != nullptr) {  //
    auto env = info.Env();
    if (info[1].IsFunction()) {
      charmods_cbs[window] = Napi::Persistent(info[1].As<Napi::Function>());
      GLFW_TRY(env, GLFWAPI::glfwSetCharModsCallback(window, GLFWcharmods_cb));
    } else {
      if (charmods_cbs.find(window) != charmods_cbs.end()) { charmods_cbs.erase(window); }
      GLFW_TRY(env, GLFWAPI::glfwSetCharModsCallback(window, NULL));
    }
  }
}

// GLFWAPI GLFWmousebuttonfun glfwSetMouseButtonCallback(GLFWwindow* window, GLFWmousebuttonfun
// callback);
void glfwSetMouseButtonCallback(Napi::CallbackInfo const& info) {
  auto window = reinterpret_cast<GLFWwindow*>(info[0].ToNumber().Int64Value());
  if (window != nullptr) {  //
    auto env = info.Env();
    if (info[1].IsFunction()) {
      mousebutton_cbs[window] = Napi::Persistent(info[1].As<Napi::Function>());
      GLFW_TRY(env, GLFWAPI::glfwSetMouseButtonCallback(window, GLFWmousebutton_cb));
    } else {
      if (mousebutton_cbs.find(window) != mousebutton_cbs.end()) { mousebutton_cbs.erase(window); }
      GLFW_TRY(env, GLFWAPI::glfwSetMouseButtonCallback(window, NULL));
    }
  }
}

void glfwSetCursorPosCallback(Napi::CallbackInfo const& info) {
  auto window = reinterpret_cast<GLFWwindow*>(info[0].ToNumber().Int64Value());
  if (window != nullptr) {  //
    auto env = info.Env();
    if (info[1].IsFunction()) {
      cursorpos_cbs[window] = Napi::Persistent(info[1].As<Napi::Function>());
      GLFW_TRY(env, GLFWAPI::glfwSetCursorPosCallback(window, GLFWcursorpos_cb));
    } else {
      if (cursorpos_cbs.find(window) != cursorpos_cbs.end()) { cursorpos_cbs.erase(window); }
      GLFW_TRY(env, GLFWAPI::glfwSetCursorPosCallback(window, NULL));
    }
  }
}

// GLFWAPI GLFWcursorenterfun glfwSetCursorEnterCallback(GLFWwindow* window, GLFWcursorenterfun
// callback);
void glfwSetCursorEnterCallback(Napi::CallbackInfo const& info) {
  auto window = reinterpret_cast<GLFWwindow*>(info[0].ToNumber().Int64Value());
  if (window != nullptr) {  //
    auto env = info.Env();
    if (info[1].IsFunction()) {
      cursorenter_cbs[window] = Napi::Persistent(info[1].As<Napi::Function>());
      GLFW_TRY(env, GLFWAPI::glfwSetCursorEnterCallback(window, GLFWcursorenter_cb));
    } else {
      if (cursorenter_cbs.find(window) != cursorenter_cbs.end()) { cursorenter_cbs.erase(window); }
      GLFW_TRY(env, GLFWAPI::glfwSetCursorEnterCallback(window, NULL));
    }
  }
}

// GLFWAPI GLFWscrollfun glfwSetScrollCallback(GLFWwindow* window, GLFWscrollfun callback);
void glfwSetScrollCallback(Napi::CallbackInfo const& info) {
  auto window = reinterpret_cast<GLFWwindow*>(info[0].ToNumber().Int64Value());
  if (window != nullptr) {  //
    auto env = info.Env();
    if (info[1].IsFunction()) {
      scroll_cbs[window] = Napi::Persistent(info[1].As<Napi::Function>());
      GLFW_TRY(env, GLFWAPI::glfwSetScrollCallback(window, GLFWscroll_cb));
    } else {
      if (scroll_cbs.find(window) != scroll_cbs.end()) { scroll_cbs.erase(window); }
      GLFW_TRY(env, GLFWAPI::glfwSetScrollCallback(window, NULL));
    }
  }
}

// GLFWAPI GLFWdropfun glfwSetDropCallback(GLFWwindow* window, GLFWdropfun callback);
void glfwSetDropCallback(Napi::CallbackInfo const& info) {
  auto window = reinterpret_cast<GLFWwindow*>(info[0].ToNumber().Int64Value());
  if (window != nullptr) {  //
    auto env = info.Env();
    if (info[1].IsFunction()) {
      drop_cbs[window] = Napi::Persistent(info[1].As<Napi::Function>());
      GLFW_TRY(env, GLFWAPI::glfwSetDropCallback(window, GLFWdrop_cb));
    } else {
      if (drop_cbs.find(window) != drop_cbs.end()) { drop_cbs.erase(window); }
      GLFW_TRY(env, GLFWAPI::glfwSetDropCallback(window, NULL));
    }
  }
}
}  // namespace nv
