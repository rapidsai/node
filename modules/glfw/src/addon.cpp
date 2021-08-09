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

#include "addon.hpp"
#include "glfw.hpp"
#include "macros.hpp"

#include <nv_node/utilities/args.hpp>
#include <nv_node/utilities/cpp_to_napi.hpp>

std::ostream& operator<<(std::ostream& os, const nv::NapiToCPP& self) {
  return os << self.operator std::string();
};

namespace nv {

// GLFWAPI int glfwInit(void);
Napi::Value glfwInit(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GLFW_EXPECT_TRUE(env, GLFWAPI::glfwInit());
  return info.This();
}

// GLFWAPI void glfwTerminate(void);
void glfwTerminate(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GLFW_TRY(env, GLFWAPI::glfwTerminate());
}

// GLFWAPI void glfwInitHint(int hint, int value);
void glfwInitHint(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  GLFW_TRY(env, GLFWAPI::glfwInitHint(args[0], args[1]));
}

// GLFWAPI void glfwGetVersion(int* major, int* minor, int* rev);
Napi::Value glfwGetVersion(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  int32_t major{}, minor{}, rev{};
  GLFW_TRY(env, GLFWAPI::glfwGetVersion(&major, &minor, &rev));
  return CPPToNapi(info)(std::map<std::string, int32_t>{//
                                                        {"major", major},
                                                        {"minor", minor},
                                                        {"rev", rev}});
}

// GLFWAPI const char* glfwGetVersionString(void);
Napi::Value glfwGetVersionString(Napi::CallbackInfo const& info) {
  auto version = GLFWAPI::glfwGetVersionString();
  return CPPToNapi(info)(std::string{version == nullptr ? "" : version});
}

// GLFWAPI int glfwGetError(const char** description);
Napi::Value glfwGetError(Napi::CallbackInfo const& info) {
  auto env        = info.Env();
  const char* err = nullptr;
  const int code  = GLFWAPI::glfwGetError(&err);
  if (code == GLFW_NO_ERROR) { return env.Undefined(); }
  return nv::glfwError(env, code, err, __FILE__, __LINE__).Value();
}

// GLFWAPI const char* glfwGetKeyName(int key, int scancode);
Napi::Value glfwGetKeyName(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  auto name = GLFWAPI::glfwGetKeyName(args[0], args[1]);
  return CPPToNapi(info)(std::string{name == nullptr ? "" : name});
}

// GLFWAPI int glfwGetKeyScancode(int key);
Napi::Value glfwGetKeyScancode(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  return CPPToNapi(info)(GLFWAPI::glfwGetKeyScancode(args[0]));
}

// GLFWAPI double glfwGetTime(void);
Napi::Value glfwGetTime(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(GLFWAPI::glfwGetTime());
}

// GLFWAPI void glfwSetTime(double time);
void glfwSetTime(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  GLFW_TRY(env, GLFWAPI::glfwSetTime(args[0]));
}

// GLFWAPI uint64_t glfwGetTimerValue(void);
Napi::Value glfwGetTimerValue(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(GLFWAPI::glfwGetTimerValue());
}

// GLFWAPI uint64_t glfwGetTimerFrequency(void);
Napi::Value glfwGetTimerFrequency(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(GLFWAPI::glfwGetTimerFrequency());
}

// GLFWAPI void glfwSwapInterval(int interval);
void glfwSwapInterval(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  GLFW_TRY(env, GLFWAPI::glfwSwapInterval(args[0]));
}

// GLFWAPI int glfwExtensionSupported(const char* extension);
Napi::Value glfwExtensionSupported(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  std::string extension       = args[0];
  auto is_extension_supported = GLFWAPI::glfwExtensionSupported(extension.data());
  return CPPToNapi(info)(static_cast<bool>(is_extension_supported));
}

// GLFWAPI GLFWglproc glfwGetProcAddress(const char* procname);
Napi::Value glfwGetProcAddress(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  std::string name = args[0];
  auto addr        = GLFWAPI::glfwGetProcAddress(name.data());
  return CPPToNapi(info)(reinterpret_cast<char*>(addr));
}

// GLFWAPI const char** glfwGetRequiredInstanceExtensions(uint32_t* count);
Napi::Value glfwGetRequiredInstanceExtensions(Napi::CallbackInfo const& info) {
  uint32_t count{};
  const char** exts = GLFWAPI::glfwGetRequiredInstanceExtensions(&count);
  return CPPToNapi(info)(std::vector<std::string>{exts, exts + count});
}

}  // namespace nv

Napi::Object initModule(Napi::Env env, Napi::Object exports) {
  EXPORT_FUNC(env, exports, "init", nv::glfwInit);
  EXPORT_FUNC(env, exports, "terminate", nv::glfwTerminate);
  EXPORT_FUNC(env, exports, "initHint", nv::glfwInitHint);
  EXPORT_FUNC(env, exports, "getVersion", nv::glfwGetVersion);
  EXPORT_FUNC(env, exports, "getVersionString", nv::glfwGetVersionString);
  EXPORT_FUNC(env, exports, "setErrorCallback", nv::glfwSetErrorCallback);
  EXPORT_FUNC(env, exports, "getMonitors", nv::glfwGetMonitors);
  EXPORT_FUNC(env, exports, "getPrimaryMonitor", nv::glfwGetPrimaryMonitor);
  EXPORT_FUNC(env, exports, "getMonitorPos", nv::glfwGetMonitorPos);
  EXPORT_FUNC(env, exports, "getMonitorWorkarea", nv::glfwGetMonitorWorkarea);
  EXPORT_FUNC(env, exports, "getMonitorPhysicalSize", nv::glfwGetMonitorPhysicalSize);
  EXPORT_FUNC(env, exports, "getMonitorContentScale", nv::glfwGetMonitorContentScale);
  EXPORT_FUNC(env, exports, "getMonitorName", nv::glfwGetMonitorName);
  EXPORT_FUNC(env, exports, "setMonitorCallback", nv::glfwSetMonitorCallback);
  EXPORT_FUNC(env, exports, "getVideoModes", nv::glfwGetVideoModes);
  EXPORT_FUNC(env, exports, "getVideoMode", nv::glfwGetVideoMode);
  EXPORT_FUNC(env, exports, "setGamma", nv::glfwSetGamma);
  EXPORT_FUNC(env, exports, "getGammaRamp", nv::glfwGetGammaRamp);
  EXPORT_FUNC(env, exports, "setGammaRamp", nv::glfwSetGammaRamp);
  EXPORT_FUNC(env, exports, "defaultWindowHints", nv::glfwDefaultWindowHints);
  EXPORT_FUNC(env, exports, "windowHint", nv::glfwWindowHint);
  EXPORT_FUNC(env, exports, "windowHintString", nv::glfwWindowHintString);
  EXPORT_FUNC(env, exports, "createWindow", nv::glfwCreateWindow);
  // #ifdef __linux__
  //   EXPORT_FUNC(env, exports, "reparentWindow", nv::glfwReparentWindow);
  // #endif
  EXPORT_FUNC(env, exports, "destroyWindow", nv::glfwDestroyWindow);
  EXPORT_FUNC(env, exports, "windowShouldClose", nv::glfwWindowShouldClose);
  EXPORT_FUNC(env, exports, "setWindowShouldClose", nv::glfwSetWindowShouldClose);
  EXPORT_FUNC(env, exports, "setWindowTitle", nv::glfwSetWindowTitle);
  EXPORT_FUNC(env, exports, "setWindowIcon", nv::glfwSetWindowIcon);
  EXPORT_FUNC(env, exports, "getWindowPos", nv::glfwGetWindowPos);
  EXPORT_FUNC(env, exports, "setWindowPos", nv::glfwSetWindowPos);
  EXPORT_FUNC(env, exports, "getWindowSize", nv::glfwGetWindowSize);
  EXPORT_FUNC(env, exports, "setWindowSizeLimits", nv::glfwSetWindowSizeLimits);
  EXPORT_FUNC(env, exports, "setWindowAspectRatio", nv::glfwSetWindowAspectRatio);
  EXPORT_FUNC(env, exports, "setWindowSize", nv::glfwSetWindowSize);
  EXPORT_FUNC(env, exports, "getFramebufferSize", nv::glfwGetFramebufferSize);
  EXPORT_FUNC(env, exports, "getWindowFrameSize", nv::glfwGetWindowFrameSize);
  EXPORT_FUNC(env, exports, "getWindowContentScale", nv::glfwGetWindowContentScale);
  EXPORT_FUNC(env, exports, "getWindowOpacity", nv::glfwGetWindowOpacity);
  EXPORT_FUNC(env, exports, "setWindowOpacity", nv::glfwSetWindowOpacity);
  EXPORT_FUNC(env, exports, "iconifyWindow", nv::glfwIconifyWindow);
  EXPORT_FUNC(env, exports, "restoreWindow", nv::glfwRestoreWindow);
  EXPORT_FUNC(env, exports, "maximizeWindow", nv::glfwMaximizeWindow);
  EXPORT_FUNC(env, exports, "showWindow", nv::glfwShowWindow);
  EXPORT_FUNC(env, exports, "hideWindow", nv::glfwHideWindow);
  EXPORT_FUNC(env, exports, "focusWindow", nv::glfwFocusWindow);
  EXPORT_FUNC(env, exports, "requestWindowAttention", nv::glfwRequestWindowAttention);
  EXPORT_FUNC(env, exports, "getWindowMonitor", nv::glfwGetWindowMonitor);
  EXPORT_FUNC(env, exports, "setWindowMonitor", nv::glfwSetWindowMonitor);
  EXPORT_FUNC(env, exports, "getWindowAttrib", nv::glfwGetWindowAttrib);
  EXPORT_FUNC(env, exports, "setWindowAttrib", nv::glfwSetWindowAttrib);
  EXPORT_FUNC(env, exports, "setWindowPosCallback", nv::glfwSetWindowPosCallback);
  EXPORT_FUNC(env, exports, "setWindowSizeCallback", nv::glfwSetWindowSizeCallback);
  EXPORT_FUNC(env, exports, "setWindowCloseCallback", nv::glfwSetWindowCloseCallback);
  EXPORT_FUNC(env, exports, "setWindowRefreshCallback", nv::glfwSetWindowRefreshCallback);
  EXPORT_FUNC(env, exports, "setWindowFocusCallback", nv::glfwSetWindowFocusCallback);
  EXPORT_FUNC(env, exports, "setWindowIconifyCallback", nv::glfwSetWindowIconifyCallback);
  EXPORT_FUNC(env, exports, "setWindowMaximizeCallback", nv::glfwSetWindowMaximizeCallback);
  EXPORT_FUNC(env, exports, "setFramebufferSizeCallback", nv::glfwSetFramebufferSizeCallback);
  EXPORT_FUNC(env, exports, "setWindowContentScaleCallback", nv::glfwSetWindowContentScaleCallback);
  EXPORT_FUNC(env, exports, "pollEvents", nv::glfwPollEvents);
  EXPORT_FUNC(env, exports, "waitEvents", nv::glfwWaitEvents);
  EXPORT_FUNC(env, exports, "waitEventsTimeout", nv::glfwWaitEventsTimeout);
  EXPORT_FUNC(env, exports, "postEmptyEvent", nv::glfwPostEmptyEvent);
  EXPORT_FUNC(env, exports, "getInputMode", nv::glfwGetInputMode);
  EXPORT_FUNC(env, exports, "setInputMode", nv::glfwSetInputMode);
  EXPORT_FUNC(env, exports, "rawMouseMotionSupported", nv::glfwRawMouseMotionSupported);
  EXPORT_FUNC(env, exports, "getKeyName", nv::glfwGetKeyName);
  EXPORT_FUNC(env, exports, "getKeyScancode", nv::glfwGetKeyScancode);
  EXPORT_FUNC(env, exports, "getKey", nv::glfwGetKey);
  EXPORT_FUNC(env, exports, "getMouseButton", nv::glfwGetMouseButton);
  EXPORT_FUNC(env, exports, "getCursorPos", nv::glfwGetCursorPos);
  EXPORT_FUNC(env, exports, "setCursorPos", nv::glfwSetCursorPos);
  EXPORT_FUNC(env, exports, "createCursor", nv::glfwCreateCursor);
  EXPORT_FUNC(env, exports, "createStandardCursor", nv::glfwCreateStandardCursor);
  EXPORT_FUNC(env, exports, "destroyCursor", nv::glfwDestroyCursor);
  EXPORT_FUNC(env, exports, "setCursor", nv::glfwSetCursor);
  EXPORT_FUNC(env, exports, "setKeyCallback", nv::glfwSetKeyCallback);
  EXPORT_FUNC(env, exports, "setCharCallback", nv::glfwSetCharCallback);
  EXPORT_FUNC(env, exports, "setCharModsCallback", nv::glfwSetCharModsCallback);
  EXPORT_FUNC(env, exports, "setMouseButtonCallback", nv::glfwSetMouseButtonCallback);
  EXPORT_FUNC(env, exports, "setCursorPosCallback", nv::glfwSetCursorPosCallback);
  EXPORT_FUNC(env, exports, "setCursorEnterCallback", nv::glfwSetCursorEnterCallback);
  EXPORT_FUNC(env, exports, "setScrollCallback", nv::glfwSetScrollCallback);
  EXPORT_FUNC(env, exports, "setDropCallback", nv::glfwSetDropCallback);
  EXPORT_FUNC(env, exports, "joystickPresent", nv::glfwJoystickPresent);
  EXPORT_FUNC(env, exports, "getJoystickAxes", nv::glfwGetJoystickAxes);
  EXPORT_FUNC(env, exports, "getJoystickButtons", nv::glfwGetJoystickButtons);
  EXPORT_FUNC(env, exports, "getJoystickHats", nv::glfwGetJoystickHats);
  EXPORT_FUNC(env, exports, "getJoystickName", nv::glfwGetJoystickName);
  EXPORT_FUNC(env, exports, "getJoystickGUID", nv::glfwGetJoystickGUID);
  EXPORT_FUNC(env, exports, "joystickIsGamepad", nv::glfwJoystickIsGamepad);
  EXPORT_FUNC(env, exports, "setJoystickCallback", nv::glfwSetJoystickCallback);
  EXPORT_FUNC(env, exports, "updateGamepadMappings", nv::glfwUpdateGamepadMappings);
  EXPORT_FUNC(env, exports, "getGamepadName", nv::glfwGetGamepadName);
  EXPORT_FUNC(env, exports, "getGamepadState", nv::glfwGetGamepadState);
  EXPORT_FUNC(env, exports, "setClipboardString", nv::glfwSetClipboardString);
  EXPORT_FUNC(env, exports, "getClipboardString", nv::glfwGetClipboardString);
  EXPORT_FUNC(env, exports, "getTime", nv::glfwGetTime);
  EXPORT_FUNC(env, exports, "setTime", nv::glfwSetTime);
  EXPORT_FUNC(env, exports, "getTimerValue", nv::glfwGetTimerValue);
  EXPORT_FUNC(env, exports, "getTimerFrequency", nv::glfwGetTimerFrequency);
  EXPORT_FUNC(env, exports, "makeContextCurrent", nv::glfwMakeContextCurrent);
  EXPORT_FUNC(env, exports, "getCurrentContext", nv::glfwGetCurrentContext);
  EXPORT_FUNC(env, exports, "swapBuffers", nv::glfwSwapBuffers);
  EXPORT_FUNC(env, exports, "swapInterval", nv::glfwSwapInterval);
  EXPORT_FUNC(env, exports, "extensionSupported", nv::glfwExtensionSupported);
  EXPORT_FUNC(env, exports, "getProcAddress", nv::glfwGetProcAddress);
  EXPORT_FUNC(env, exports, "vulkanSupported", nv::glfwVulkanSupported);
  EXPORT_FUNC(env, exports, "getRequiredInstanceExtensions", nv::glfwGetRequiredInstanceExtensions);

  EXPORT_ENUM(env, exports, "VERSION_MAJOR", GLFW_VERSION_MAJOR);
  EXPORT_ENUM(env, exports, "VERSION_MINOR", GLFW_VERSION_MINOR);
  EXPORT_ENUM(env, exports, "VERSION_REVISION", GLFW_VERSION_REVISION);

  EXPORT_ENUM(env, exports, "TRUE", GLFW_TRUE);
  EXPORT_ENUM(env, exports, "FALSE", GLFW_FALSE);

  EXPORT_ENUM(env, exports, "RELEASE", GLFW_RELEASE);
  EXPORT_ENUM(env, exports, "PRESS", GLFW_PRESS);
  EXPORT_ENUM(env, exports, "REPEAT", GLFW_REPEAT);

  EXPORT_ENUM(env, exports, "HAT_CENTERED", GLFW_HAT_CENTERED);
  EXPORT_ENUM(env, exports, "HAT_UP", GLFW_HAT_UP);
  EXPORT_ENUM(env, exports, "HAT_RIGHT", GLFW_HAT_RIGHT);
  EXPORT_ENUM(env, exports, "HAT_DOWN", GLFW_HAT_DOWN);
  EXPORT_ENUM(env, exports, "HAT_LEFT", GLFW_HAT_LEFT);
  EXPORT_ENUM(env, exports, "HAT_RIGHT_UP", GLFW_HAT_RIGHT_UP);
  EXPORT_ENUM(env, exports, "HAT_RIGHT_DOWN", GLFW_HAT_RIGHT_DOWN);
  EXPORT_ENUM(env, exports, "HAT_LEFT_UP", GLFW_HAT_LEFT_UP);
  EXPORT_ENUM(env, exports, "HAT_LEFT_DOWN", GLFW_HAT_LEFT_DOWN);

  EXPORT_ENUM(env, exports, "KEY_UNKNOWN", GLFW_KEY_UNKNOWN);
  EXPORT_ENUM(env, exports, "KEY_SPACE", GLFW_KEY_SPACE);
  EXPORT_ENUM(env, exports, "KEY_APOSTROPHE", GLFW_KEY_APOSTROPHE);
  EXPORT_ENUM(env, exports, "KEY_COMMA", GLFW_KEY_COMMA);
  EXPORT_ENUM(env, exports, "KEY_MINUS", GLFW_KEY_MINUS);
  EXPORT_ENUM(env, exports, "KEY_PERIOD", GLFW_KEY_PERIOD);
  EXPORT_ENUM(env, exports, "KEY_SLASH", GLFW_KEY_SLASH);
  EXPORT_ENUM(env, exports, "KEY_0", GLFW_KEY_0);
  EXPORT_ENUM(env, exports, "KEY_1", GLFW_KEY_1);
  EXPORT_ENUM(env, exports, "KEY_2", GLFW_KEY_2);
  EXPORT_ENUM(env, exports, "KEY_3", GLFW_KEY_3);
  EXPORT_ENUM(env, exports, "KEY_4", GLFW_KEY_4);
  EXPORT_ENUM(env, exports, "KEY_5", GLFW_KEY_5);
  EXPORT_ENUM(env, exports, "KEY_6", GLFW_KEY_6);
  EXPORT_ENUM(env, exports, "KEY_7", GLFW_KEY_7);
  EXPORT_ENUM(env, exports, "KEY_8", GLFW_KEY_8);
  EXPORT_ENUM(env, exports, "KEY_9", GLFW_KEY_9);
  EXPORT_ENUM(env, exports, "KEY_SEMICOLON", GLFW_KEY_SEMICOLON);
  EXPORT_ENUM(env, exports, "KEY_EQUAL", GLFW_KEY_EQUAL);
  EXPORT_ENUM(env, exports, "KEY_A", GLFW_KEY_A);
  EXPORT_ENUM(env, exports, "KEY_B", GLFW_KEY_B);
  EXPORT_ENUM(env, exports, "KEY_C", GLFW_KEY_C);
  EXPORT_ENUM(env, exports, "KEY_D", GLFW_KEY_D);
  EXPORT_ENUM(env, exports, "KEY_E", GLFW_KEY_E);
  EXPORT_ENUM(env, exports, "KEY_F", GLFW_KEY_F);
  EXPORT_ENUM(env, exports, "KEY_G", GLFW_KEY_G);
  EXPORT_ENUM(env, exports, "KEY_H", GLFW_KEY_H);
  EXPORT_ENUM(env, exports, "KEY_I", GLFW_KEY_I);
  EXPORT_ENUM(env, exports, "KEY_J", GLFW_KEY_J);
  EXPORT_ENUM(env, exports, "KEY_K", GLFW_KEY_K);
  EXPORT_ENUM(env, exports, "KEY_L", GLFW_KEY_L);
  EXPORT_ENUM(env, exports, "KEY_M", GLFW_KEY_M);
  EXPORT_ENUM(env, exports, "KEY_N", GLFW_KEY_N);
  EXPORT_ENUM(env, exports, "KEY_O", GLFW_KEY_O);
  EXPORT_ENUM(env, exports, "KEY_P", GLFW_KEY_P);
  EXPORT_ENUM(env, exports, "KEY_Q", GLFW_KEY_Q);
  EXPORT_ENUM(env, exports, "KEY_R", GLFW_KEY_R);
  EXPORT_ENUM(env, exports, "KEY_S", GLFW_KEY_S);
  EXPORT_ENUM(env, exports, "KEY_T", GLFW_KEY_T);
  EXPORT_ENUM(env, exports, "KEY_U", GLFW_KEY_U);
  EXPORT_ENUM(env, exports, "KEY_V", GLFW_KEY_V);
  EXPORT_ENUM(env, exports, "KEY_W", GLFW_KEY_W);
  EXPORT_ENUM(env, exports, "KEY_X", GLFW_KEY_X);
  EXPORT_ENUM(env, exports, "KEY_Y", GLFW_KEY_Y);
  EXPORT_ENUM(env, exports, "KEY_Z", GLFW_KEY_Z);
  EXPORT_ENUM(env, exports, "KEY_LEFT_BRACKET", GLFW_KEY_LEFT_BRACKET);
  EXPORT_ENUM(env, exports, "KEY_BACKSLASH", GLFW_KEY_BACKSLASH);
  EXPORT_ENUM(env, exports, "KEY_RIGHT_BRACKET", GLFW_KEY_RIGHT_BRACKET);
  EXPORT_ENUM(env, exports, "KEY_GRAVE_ACCENT", GLFW_KEY_GRAVE_ACCENT);
  EXPORT_ENUM(env, exports, "KEY_WORLD_1", GLFW_KEY_WORLD_1);
  EXPORT_ENUM(env, exports, "KEY_WORLD_2", GLFW_KEY_WORLD_2);
  EXPORT_ENUM(env, exports, "KEY_ESCAPE", GLFW_KEY_ESCAPE);
  EXPORT_ENUM(env, exports, "KEY_ENTER", GLFW_KEY_ENTER);
  EXPORT_ENUM(env, exports, "KEY_TAB", GLFW_KEY_TAB);
  EXPORT_ENUM(env, exports, "KEY_BACKSPACE", GLFW_KEY_BACKSPACE);
  EXPORT_ENUM(env, exports, "KEY_INSERT", GLFW_KEY_INSERT);
  EXPORT_ENUM(env, exports, "KEY_DELETE", GLFW_KEY_DELETE);
  EXPORT_ENUM(env, exports, "KEY_RIGHT", GLFW_KEY_RIGHT);
  EXPORT_ENUM(env, exports, "KEY_LEFT", GLFW_KEY_LEFT);
  EXPORT_ENUM(env, exports, "KEY_DOWN", GLFW_KEY_DOWN);
  EXPORT_ENUM(env, exports, "KEY_UP", GLFW_KEY_UP);
  EXPORT_ENUM(env, exports, "KEY_PAGE_UP", GLFW_KEY_PAGE_UP);
  EXPORT_ENUM(env, exports, "KEY_PAGE_DOWN", GLFW_KEY_PAGE_DOWN);
  EXPORT_ENUM(env, exports, "KEY_HOME", GLFW_KEY_HOME);
  EXPORT_ENUM(env, exports, "KEY_END", GLFW_KEY_END);
  EXPORT_ENUM(env, exports, "KEY_CAPS_LOCK", GLFW_KEY_CAPS_LOCK);
  EXPORT_ENUM(env, exports, "KEY_SCROLL_LOCK", GLFW_KEY_SCROLL_LOCK);
  EXPORT_ENUM(env, exports, "KEY_NUM_LOCK", GLFW_KEY_NUM_LOCK);
  EXPORT_ENUM(env, exports, "KEY_PRINT_SCREEN", GLFW_KEY_PRINT_SCREEN);
  EXPORT_ENUM(env, exports, "KEY_PAUSE", GLFW_KEY_PAUSE);
  EXPORT_ENUM(env, exports, "KEY_F1", GLFW_KEY_F1);
  EXPORT_ENUM(env, exports, "KEY_F2", GLFW_KEY_F2);
  EXPORT_ENUM(env, exports, "KEY_F3", GLFW_KEY_F3);
  EXPORT_ENUM(env, exports, "KEY_F4", GLFW_KEY_F4);
  EXPORT_ENUM(env, exports, "KEY_F5", GLFW_KEY_F5);
  EXPORT_ENUM(env, exports, "KEY_F6", GLFW_KEY_F6);
  EXPORT_ENUM(env, exports, "KEY_F7", GLFW_KEY_F7);
  EXPORT_ENUM(env, exports, "KEY_F8", GLFW_KEY_F8);
  EXPORT_ENUM(env, exports, "KEY_F9", GLFW_KEY_F9);
  EXPORT_ENUM(env, exports, "KEY_F10", GLFW_KEY_F10);
  EXPORT_ENUM(env, exports, "KEY_F11", GLFW_KEY_F11);
  EXPORT_ENUM(env, exports, "KEY_F12", GLFW_KEY_F12);
  EXPORT_ENUM(env, exports, "KEY_F13", GLFW_KEY_F13);
  EXPORT_ENUM(env, exports, "KEY_F14", GLFW_KEY_F14);
  EXPORT_ENUM(env, exports, "KEY_F15", GLFW_KEY_F15);
  EXPORT_ENUM(env, exports, "KEY_F16", GLFW_KEY_F16);
  EXPORT_ENUM(env, exports, "KEY_F17", GLFW_KEY_F17);
  EXPORT_ENUM(env, exports, "KEY_F18", GLFW_KEY_F18);
  EXPORT_ENUM(env, exports, "KEY_F19", GLFW_KEY_F19);
  EXPORT_ENUM(env, exports, "KEY_F20", GLFW_KEY_F20);
  EXPORT_ENUM(env, exports, "KEY_F21", GLFW_KEY_F21);
  EXPORT_ENUM(env, exports, "KEY_F22", GLFW_KEY_F22);
  EXPORT_ENUM(env, exports, "KEY_F23", GLFW_KEY_F23);
  EXPORT_ENUM(env, exports, "KEY_F24", GLFW_KEY_F24);
  EXPORT_ENUM(env, exports, "KEY_F25", GLFW_KEY_F25);
  EXPORT_ENUM(env, exports, "KEY_KP_0", GLFW_KEY_KP_0);
  EXPORT_ENUM(env, exports, "KEY_KP_1", GLFW_KEY_KP_1);
  EXPORT_ENUM(env, exports, "KEY_KP_2", GLFW_KEY_KP_2);
  EXPORT_ENUM(env, exports, "KEY_KP_3", GLFW_KEY_KP_3);
  EXPORT_ENUM(env, exports, "KEY_KP_4", GLFW_KEY_KP_4);
  EXPORT_ENUM(env, exports, "KEY_KP_5", GLFW_KEY_KP_5);
  EXPORT_ENUM(env, exports, "KEY_KP_6", GLFW_KEY_KP_6);
  EXPORT_ENUM(env, exports, "KEY_KP_7", GLFW_KEY_KP_7);
  EXPORT_ENUM(env, exports, "KEY_KP_8", GLFW_KEY_KP_8);
  EXPORT_ENUM(env, exports, "KEY_KP_9", GLFW_KEY_KP_9);
  EXPORT_ENUM(env, exports, "KEY_KP_DECIMAL", GLFW_KEY_KP_DECIMAL);
  EXPORT_ENUM(env, exports, "KEY_KP_DIVIDE", GLFW_KEY_KP_DIVIDE);
  EXPORT_ENUM(env, exports, "KEY_KP_MULTIPLY", GLFW_KEY_KP_MULTIPLY);
  EXPORT_ENUM(env, exports, "KEY_KP_SUBTRACT", GLFW_KEY_KP_SUBTRACT);
  EXPORT_ENUM(env, exports, "KEY_KP_ADD", GLFW_KEY_KP_ADD);
  EXPORT_ENUM(env, exports, "KEY_KP_ENTER", GLFW_KEY_KP_ENTER);
  EXPORT_ENUM(env, exports, "KEY_KP_EQUAL", GLFW_KEY_KP_EQUAL);
  EXPORT_ENUM(env, exports, "KEY_LEFT_SHIFT", GLFW_KEY_LEFT_SHIFT);
  EXPORT_ENUM(env, exports, "KEY_LEFT_CONTROL", GLFW_KEY_LEFT_CONTROL);
  EXPORT_ENUM(env, exports, "KEY_LEFT_ALT", GLFW_KEY_LEFT_ALT);
  EXPORT_ENUM(env, exports, "KEY_LEFT_SUPER", GLFW_KEY_LEFT_SUPER);
  EXPORT_ENUM(env, exports, "KEY_RIGHT_SHIFT", GLFW_KEY_RIGHT_SHIFT);
  EXPORT_ENUM(env, exports, "KEY_RIGHT_CONTROL", GLFW_KEY_RIGHT_CONTROL);
  EXPORT_ENUM(env, exports, "KEY_RIGHT_ALT", GLFW_KEY_RIGHT_ALT);
  EXPORT_ENUM(env, exports, "KEY_RIGHT_SUPER", GLFW_KEY_RIGHT_SUPER);
  EXPORT_ENUM(env, exports, "KEY_MENU", GLFW_KEY_MENU);
  EXPORT_ENUM(env, exports, "KEY_LAST", GLFW_KEY_LAST);

  EXPORT_ENUM(env, exports, "MOD_SHIFT", GLFW_MOD_SHIFT);
  EXPORT_ENUM(env, exports, "MOD_CONTROL", GLFW_MOD_CONTROL);
  EXPORT_ENUM(env, exports, "MOD_ALT", GLFW_MOD_ALT);
  EXPORT_ENUM(env, exports, "MOD_SUPER", GLFW_MOD_SUPER);
  EXPORT_ENUM(env, exports, "MOD_CAPS_LOCK", GLFW_MOD_CAPS_LOCK);
  EXPORT_ENUM(env, exports, "MOD_NUM_LOCK", GLFW_MOD_NUM_LOCK);

  EXPORT_ENUM(env, exports, "MOUSE_BUTTON_1", GLFW_MOUSE_BUTTON_1);
  EXPORT_ENUM(env, exports, "MOUSE_BUTTON_2", GLFW_MOUSE_BUTTON_2);
  EXPORT_ENUM(env, exports, "MOUSE_BUTTON_3", GLFW_MOUSE_BUTTON_3);
  EXPORT_ENUM(env, exports, "MOUSE_BUTTON_4", GLFW_MOUSE_BUTTON_4);
  EXPORT_ENUM(env, exports, "MOUSE_BUTTON_5", GLFW_MOUSE_BUTTON_5);
  EXPORT_ENUM(env, exports, "MOUSE_BUTTON_6", GLFW_MOUSE_BUTTON_6);
  EXPORT_ENUM(env, exports, "MOUSE_BUTTON_7", GLFW_MOUSE_BUTTON_7);
  EXPORT_ENUM(env, exports, "MOUSE_BUTTON_8", GLFW_MOUSE_BUTTON_8);
  EXPORT_ENUM(env, exports, "MOUSE_BUTTON_LAST", GLFW_MOUSE_BUTTON_LAST);
  EXPORT_ENUM(env, exports, "MOUSE_BUTTON_LEFT", GLFW_MOUSE_BUTTON_LEFT);
  EXPORT_ENUM(env, exports, "MOUSE_BUTTON_RIGHT", GLFW_MOUSE_BUTTON_RIGHT);
  EXPORT_ENUM(env, exports, "MOUSE_BUTTON_MIDDLE", GLFW_MOUSE_BUTTON_MIDDLE);

  EXPORT_ENUM(env, exports, "JOYSTICK_1", GLFW_JOYSTICK_1);
  EXPORT_ENUM(env, exports, "JOYSTICK_2", GLFW_JOYSTICK_2);
  EXPORT_ENUM(env, exports, "JOYSTICK_3", GLFW_JOYSTICK_3);
  EXPORT_ENUM(env, exports, "JOYSTICK_4", GLFW_JOYSTICK_4);
  EXPORT_ENUM(env, exports, "JOYSTICK_5", GLFW_JOYSTICK_5);
  EXPORT_ENUM(env, exports, "JOYSTICK_6", GLFW_JOYSTICK_6);
  EXPORT_ENUM(env, exports, "JOYSTICK_7", GLFW_JOYSTICK_7);
  EXPORT_ENUM(env, exports, "JOYSTICK_8", GLFW_JOYSTICK_8);
  EXPORT_ENUM(env, exports, "JOYSTICK_9", GLFW_JOYSTICK_9);
  EXPORT_ENUM(env, exports, "JOYSTICK_10", GLFW_JOYSTICK_10);
  EXPORT_ENUM(env, exports, "JOYSTICK_11", GLFW_JOYSTICK_11);
  EXPORT_ENUM(env, exports, "JOYSTICK_12", GLFW_JOYSTICK_12);
  EXPORT_ENUM(env, exports, "JOYSTICK_13", GLFW_JOYSTICK_13);
  EXPORT_ENUM(env, exports, "JOYSTICK_14", GLFW_JOYSTICK_14);
  EXPORT_ENUM(env, exports, "JOYSTICK_15", GLFW_JOYSTICK_15);
  EXPORT_ENUM(env, exports, "JOYSTICK_16", GLFW_JOYSTICK_16);
  EXPORT_ENUM(env, exports, "JOYSTICK_LAST", GLFW_JOYSTICK_LAST);

  EXPORT_ENUM(env, exports, "GAMEPAD_BUTTON_A", GLFW_GAMEPAD_BUTTON_A);
  EXPORT_ENUM(env, exports, "GAMEPAD_BUTTON_B", GLFW_GAMEPAD_BUTTON_B);
  EXPORT_ENUM(env, exports, "GAMEPAD_BUTTON_X", GLFW_GAMEPAD_BUTTON_X);
  EXPORT_ENUM(env, exports, "GAMEPAD_BUTTON_Y", GLFW_GAMEPAD_BUTTON_Y);
  EXPORT_ENUM(env, exports, "GAMEPAD_BUTTON_LEFT_BUMPER", GLFW_GAMEPAD_BUTTON_LEFT_BUMPER);
  EXPORT_ENUM(env, exports, "GAMEPAD_BUTTON_RIGHT_BUMPER", GLFW_GAMEPAD_BUTTON_RIGHT_BUMPER);
  EXPORT_ENUM(env, exports, "GAMEPAD_BUTTON_BACK", GLFW_GAMEPAD_BUTTON_BACK);
  EXPORT_ENUM(env, exports, "GAMEPAD_BUTTON_START", GLFW_GAMEPAD_BUTTON_START);
  EXPORT_ENUM(env, exports, "GAMEPAD_BUTTON_GUIDE", GLFW_GAMEPAD_BUTTON_GUIDE);
  EXPORT_ENUM(env, exports, "GAMEPAD_BUTTON_LEFT_THUMB", GLFW_GAMEPAD_BUTTON_LEFT_THUMB);
  EXPORT_ENUM(env, exports, "GAMEPAD_BUTTON_RIGHT_THUMB", GLFW_GAMEPAD_BUTTON_RIGHT_THUMB);
  EXPORT_ENUM(env, exports, "GAMEPAD_BUTTON_DPAD_UP", GLFW_GAMEPAD_BUTTON_DPAD_UP);
  EXPORT_ENUM(env, exports, "GAMEPAD_BUTTON_DPAD_RIGHT", GLFW_GAMEPAD_BUTTON_DPAD_RIGHT);
  EXPORT_ENUM(env, exports, "GAMEPAD_BUTTON_DPAD_DOWN", GLFW_GAMEPAD_BUTTON_DPAD_DOWN);
  EXPORT_ENUM(env, exports, "GAMEPAD_BUTTON_DPAD_LEFT", GLFW_GAMEPAD_BUTTON_DPAD_LEFT);
  EXPORT_ENUM(env, exports, "GAMEPAD_BUTTON_LAST", GLFW_GAMEPAD_BUTTON_LAST);
  EXPORT_ENUM(env, exports, "GAMEPAD_BUTTON_CROSS", GLFW_GAMEPAD_BUTTON_CROSS);
  EXPORT_ENUM(env, exports, "GAMEPAD_BUTTON_CIRCLE", GLFW_GAMEPAD_BUTTON_CIRCLE);
  EXPORT_ENUM(env, exports, "GAMEPAD_BUTTON_SQUARE", GLFW_GAMEPAD_BUTTON_SQUARE);
  EXPORT_ENUM(env, exports, "GAMEPAD_BUTTON_TRIANGLE", GLFW_GAMEPAD_BUTTON_TRIANGLE);
  EXPORT_ENUM(env, exports, "GAMEPAD_AXIS_LEFT_X", GLFW_GAMEPAD_AXIS_LEFT_X);
  EXPORT_ENUM(env, exports, "GAMEPAD_AXIS_LEFT_Y", GLFW_GAMEPAD_AXIS_LEFT_Y);
  EXPORT_ENUM(env, exports, "GAMEPAD_AXIS_RIGHT_X", GLFW_GAMEPAD_AXIS_RIGHT_X);
  EXPORT_ENUM(env, exports, "GAMEPAD_AXIS_RIGHT_Y", GLFW_GAMEPAD_AXIS_RIGHT_Y);
  EXPORT_ENUM(env, exports, "GAMEPAD_AXIS_LEFT_TRIGGER", GLFW_GAMEPAD_AXIS_LEFT_TRIGGER);
  EXPORT_ENUM(env, exports, "GAMEPAD_AXIS_RIGHT_TRIGGER", GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER);
  EXPORT_ENUM(env, exports, "GAMEPAD_AXIS_LAST", GLFW_GAMEPAD_AXIS_LAST);

  EXPORT_ENUM(env, exports, "NO_ERROR", GLFW_NO_ERROR);
  EXPORT_ENUM(env, exports, "NOT_INITIALIZED", GLFW_NOT_INITIALIZED);
  EXPORT_ENUM(env, exports, "NO_CURRENT_CONTEXT", GLFW_NO_CURRENT_CONTEXT);
  EXPORT_ENUM(env, exports, "INVALID_ENUM", GLFW_INVALID_ENUM);
  EXPORT_ENUM(env, exports, "INVALID_VALUE", GLFW_INVALID_VALUE);
  EXPORT_ENUM(env, exports, "OUT_OF_MEMORY", GLFW_OUT_OF_MEMORY);
  EXPORT_ENUM(env, exports, "API_UNAVAILABLE", GLFW_API_UNAVAILABLE);
  EXPORT_ENUM(env, exports, "VERSION_UNAVAILABLE", GLFW_VERSION_UNAVAILABLE);
  EXPORT_ENUM(env, exports, "PLATFORM_ERROR", GLFW_PLATFORM_ERROR);
  EXPORT_ENUM(env, exports, "FORMAT_UNAVAILABLE", GLFW_FORMAT_UNAVAILABLE);
  EXPORT_ENUM(env, exports, "NO_WINDOW_CONTEXT", GLFW_NO_WINDOW_CONTEXT);

  EXPORT_ENUM(env, exports, "FOCUSED", GLFW_FOCUSED);
  EXPORT_ENUM(env, exports, "ICONIFIED", GLFW_ICONIFIED);
  EXPORT_ENUM(env, exports, "RESIZABLE", GLFW_RESIZABLE);
  EXPORT_ENUM(env, exports, "VISIBLE", GLFW_VISIBLE);
  EXPORT_ENUM(env, exports, "DECORATED", GLFW_DECORATED);
  EXPORT_ENUM(env, exports, "AUTO_ICONIFY", GLFW_AUTO_ICONIFY);
  EXPORT_ENUM(env, exports, "FLOATING", GLFW_FLOATING);
  EXPORT_ENUM(env, exports, "MAXIMIZED", GLFW_MAXIMIZED);
  EXPORT_ENUM(env, exports, "CENTER_CURSOR", GLFW_CENTER_CURSOR);
  EXPORT_ENUM(env, exports, "TRANSPARENT_FRAMEBUFFER", GLFW_TRANSPARENT_FRAMEBUFFER);

  EXPORT_ENUM(env, exports, "HOVERED", GLFW_HOVERED);
  EXPORT_ENUM(env, exports, "FOCUS_ON_SHOW", GLFW_FOCUS_ON_SHOW);
  EXPORT_ENUM(env, exports, "RED_BITS", GLFW_RED_BITS);
  EXPORT_ENUM(env, exports, "GREEN_BITS", GLFW_GREEN_BITS);
  EXPORT_ENUM(env, exports, "BLUE_BITS", GLFW_BLUE_BITS);
  EXPORT_ENUM(env, exports, "ALPHA_BITS", GLFW_ALPHA_BITS);
  EXPORT_ENUM(env, exports, "DEPTH_BITS", GLFW_DEPTH_BITS);
  EXPORT_ENUM(env, exports, "STENCIL_BITS", GLFW_STENCIL_BITS);
  EXPORT_ENUM(env, exports, "ACCUM_RED_BITS", GLFW_ACCUM_RED_BITS);
  EXPORT_ENUM(env, exports, "ACCUM_GREEN_BITS", GLFW_ACCUM_GREEN_BITS);
  EXPORT_ENUM(env, exports, "ACCUM_BLUE_BITS", GLFW_ACCUM_BLUE_BITS);
  EXPORT_ENUM(env, exports, "ACCUM_ALPHA_BITS", GLFW_ACCUM_ALPHA_BITS);
  EXPORT_ENUM(env, exports, "AUX_BUFFERS", GLFW_AUX_BUFFERS);
  EXPORT_ENUM(env, exports, "STEREO", GLFW_STEREO);
  EXPORT_ENUM(env, exports, "SAMPLES", GLFW_SAMPLES);
  EXPORT_ENUM(env, exports, "SRGB_CAPABLE", GLFW_SRGB_CAPABLE);
  EXPORT_ENUM(env, exports, "REFRESH_RATE", GLFW_REFRESH_RATE);
  EXPORT_ENUM(env, exports, "DOUBLEBUFFER", GLFW_DOUBLEBUFFER);

  EXPORT_ENUM(env, exports, "CLIENT_API", GLFW_CLIENT_API);
  EXPORT_ENUM(env, exports, "CONTEXT_VERSION_MAJOR", GLFW_CONTEXT_VERSION_MAJOR);
  EXPORT_ENUM(env, exports, "CONTEXT_VERSION_MINOR", GLFW_CONTEXT_VERSION_MINOR);
  EXPORT_ENUM(env, exports, "CONTEXT_REVISION", GLFW_CONTEXT_REVISION);
  EXPORT_ENUM(env, exports, "CONTEXT_ROBUSTNESS", GLFW_CONTEXT_ROBUSTNESS);
  EXPORT_ENUM(env, exports, "OPENGL_FORWARD_COMPAT", GLFW_OPENGL_FORWARD_COMPAT);
  EXPORT_ENUM(env, exports, "OPENGL_DEBUG_CONTEXT", GLFW_OPENGL_DEBUG_CONTEXT);
  EXPORT_ENUM(env, exports, "OPENGL_PROFILE", GLFW_OPENGL_PROFILE);
  EXPORT_ENUM(env, exports, "CONTEXT_RELEASE_BEHAVIOR", GLFW_CONTEXT_RELEASE_BEHAVIOR);
  EXPORT_ENUM(env, exports, "CONTEXT_NO_ERROR", GLFW_CONTEXT_NO_ERROR);
  EXPORT_ENUM(env, exports, "CONTEXT_CREATION_API", GLFW_CONTEXT_CREATION_API);
  EXPORT_ENUM(env, exports, "SCALE_TO_MONITOR", GLFW_SCALE_TO_MONITOR);
  EXPORT_ENUM(env, exports, "COCOA_RETINA_FRAMEBUFFER", GLFW_COCOA_RETINA_FRAMEBUFFER);
  EXPORT_ENUM(env, exports, "COCOA_FRAME_NAME", GLFW_COCOA_FRAME_NAME);
  EXPORT_ENUM(env, exports, "COCOA_GRAPHICS_SWITCHING", GLFW_COCOA_GRAPHICS_SWITCHING);
  EXPORT_ENUM(env, exports, "X11_CLASS_NAME", GLFW_X11_CLASS_NAME);
  EXPORT_ENUM(env, exports, "X11_INSTANCE_NAME", GLFW_X11_INSTANCE_NAME);

  EXPORT_ENUM(env, exports, "NO_API", GLFW_NO_API);
  EXPORT_ENUM(env, exports, "OPENGL_API", GLFW_OPENGL_API);
  EXPORT_ENUM(env, exports, "OPENGL_ES_API", GLFW_OPENGL_ES_API);

  EXPORT_ENUM(env, exports, "NO_ROBUSTNESS", GLFW_NO_ROBUSTNESS);
  EXPORT_ENUM(env, exports, "NO_RESET_NOTIFICATION", GLFW_NO_RESET_NOTIFICATION);
  EXPORT_ENUM(env, exports, "LOSE_CONTEXT_ON_RESET", GLFW_LOSE_CONTEXT_ON_RESET);
  EXPORT_ENUM(env, exports, "OPENGL_ANY_PROFILE", GLFW_OPENGL_ANY_PROFILE);
  EXPORT_ENUM(env, exports, "OPENGL_CORE_PROFILE", GLFW_OPENGL_CORE_PROFILE);
  EXPORT_ENUM(env, exports, "OPENGL_COMPAT_PROFILE", GLFW_OPENGL_COMPAT_PROFILE);

  EXPORT_ENUM(env, exports, "CURSOR", GLFW_CURSOR);

  EXPORT_ENUM(env, exports, "STICKY_KEYS", GLFW_STICKY_KEYS);
  EXPORT_ENUM(env, exports, "STICKY_MOUSE_BUTTONS", GLFW_STICKY_MOUSE_BUTTONS);
  EXPORT_ENUM(env, exports, "LOCK_KEY_MODS", GLFW_LOCK_KEY_MODS);

  EXPORT_ENUM(env, exports, "RAW_MOUSE_MOTION", GLFW_RAW_MOUSE_MOTION);

  EXPORT_ENUM(env, exports, "CURSOR_NORMAL", GLFW_CURSOR_NORMAL);
  EXPORT_ENUM(env, exports, "CURSOR_HIDDEN", GLFW_CURSOR_HIDDEN);
  EXPORT_ENUM(env, exports, "CURSOR_DISABLED", GLFW_CURSOR_DISABLED);
  EXPORT_ENUM(env, exports, "ANY_RELEASE_BEHAVIOR", GLFW_ANY_RELEASE_BEHAVIOR);
  EXPORT_ENUM(env, exports, "RELEASE_BEHAVIOR_FLUSH", GLFW_RELEASE_BEHAVIOR_FLUSH);
  EXPORT_ENUM(env, exports, "RELEASE_BEHAVIOR_NONE", GLFW_RELEASE_BEHAVIOR_NONE);
  EXPORT_ENUM(env, exports, "NATIVE_CONTEXT_API", GLFW_NATIVE_CONTEXT_API);
  EXPORT_ENUM(env, exports, "EGL_CONTEXT_API", GLFW_EGL_CONTEXT_API);
  EXPORT_ENUM(env, exports, "OSMESA_CONTEXT_API", GLFW_OSMESA_CONTEXT_API);
  EXPORT_ENUM(env, exports, "ARROW_CURSOR", GLFW_ARROW_CURSOR);
  EXPORT_ENUM(env, exports, "IBEAM_CURSOR", GLFW_IBEAM_CURSOR);
  EXPORT_ENUM(env, exports, "CROSSHAIR_CURSOR", GLFW_CROSSHAIR_CURSOR);
  EXPORT_ENUM(env, exports, "HAND_CURSOR", GLFW_HAND_CURSOR);
  EXPORT_ENUM(env, exports, "HRESIZE_CURSOR", GLFW_HRESIZE_CURSOR);
  EXPORT_ENUM(env, exports, "VRESIZE_CURSOR", GLFW_VRESIZE_CURSOR);

  EXPORT_ENUM(env, exports, "CONNECTED", GLFW_CONNECTED);
  EXPORT_ENUM(env, exports, "DISCONNECTED", GLFW_DISCONNECTED);

  EXPORT_ENUM(env, exports, "JOYSTICK_HAT_BUTTONS", GLFW_JOYSTICK_HAT_BUTTONS);

  EXPORT_ENUM(env, exports, "COCOA_CHDIR_RESOURCES", GLFW_COCOA_CHDIR_RESOURCES);
  EXPORT_ENUM(env, exports, "COCOA_MENUBAR", GLFW_COCOA_MENUBAR);

  EXPORT_ENUM(env, exports, "DONT_CARE", GLFW_DONT_CARE);

  return exports;
}

NODE_API_MODULE(nv, initModule);
