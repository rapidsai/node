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

#pragma once

#include <napi.h>
#include "glfw.hpp"

namespace nv {

// GLFWAPI int glfwInit(void);
Napi::Value glfwInit(Napi::CallbackInfo const& info);

// GLFWAPI void glfwTerminate(void);
void glfwTerminate(Napi::CallbackInfo const& info);

// GLFWAPI void glfwInitHint(int hint, int value);
void glfwInitHint(Napi::CallbackInfo const& info);

// GLFWAPI void glfwGetVersion(int* major, int* minor, int* rev);
Napi::Value glfwGetVersion(Napi::CallbackInfo const& info);

// GLFWAPI const char* glfwGetVersionString(void);
Napi::Value glfwGetVersionString(Napi::CallbackInfo const& info);

// GLFWAPI int glfwGetError(const char** description);
Napi::Value glfwGetError(Napi::CallbackInfo const& info);

// GLFWAPI GLFWerrorfun glfwSetErrorCallback(GLFWerrorfun callback);
void glfwSetErrorCallback(Napi::CallbackInfo const& info);

// GLFWAPI GLFWmonitor** glfwGetMonitors(int* count);
Napi::Value glfwGetMonitors(Napi::CallbackInfo const& info);

// GLFWAPI GLFWmonitor* glfwGetPrimaryMonitor(void);
Napi::Value glfwGetPrimaryMonitor(Napi::CallbackInfo const& info);

// GLFWAPI void glfwGetMonitorPos(GLFWmonitor* monitor, int* xpos, int* ypos);
Napi::Value glfwGetMonitorPos(Napi::CallbackInfo const& info);

// GLFWAPI void glfwGetMonitorWorkarea(GLFWmonitor* monitor, int* xpos, int* ypos, int* width, int*
// height);
Napi::Value glfwGetMonitorWorkarea(Napi::CallbackInfo const& info);

// GLFWAPI void glfwGetMonitorPhysicalSize(GLFWmonitor* monitor, int* widthMM, int* heightMM);
Napi::Value glfwGetMonitorPhysicalSize(Napi::CallbackInfo const& info);

// GLFWAPI void glfwGetMonitorContentScale(GLFWmonitor* monitor, float* xscale, float* yscale);
Napi::Value glfwGetMonitorContentScale(Napi::CallbackInfo const& info);

// GLFWAPI const char* glfwGetMonitorName(GLFWmonitor* monitor);
Napi::Value glfwGetMonitorName(Napi::CallbackInfo const& info);

// // GLFWAPI void glfwSetMonitorUserPointer(GLFWmonitor* monitor, void* pointer);
// void glfwSetMonitorUserPointer(Napi::CallbackInfo const& info);

// // GLFWAPI void* glfwGetMonitorUserPointer(GLFWmonitor* monitor);
// Napi::Value glfwGetMonitorUserPointer(Napi::CallbackInfo const& info);

// GLFWAPI GLFWmonitorfun glfwSetMonitorCallback(GLFWmonitorfun callback);
void glfwSetMonitorCallback(Napi::CallbackInfo const& info);

// GLFWAPI const GLFWvidmode* glfwGetVideoModes(GLFWmonitor* monitor, int* count);
Napi::Value glfwGetVideoModes(Napi::CallbackInfo const& info);

// GLFWAPI const GLFWvidmode* glfwGetVideoMode(GLFWmonitor* monitor);
Napi::Value glfwGetVideoMode(Napi::CallbackInfo const& info);

// GLFWAPI void glfwSetGamma(GLFWmonitor* monitor, float gamma);
void glfwSetGamma(Napi::CallbackInfo const& info);

// GLFWAPI const GLFWgammaramp* glfwGetGammaRamp(GLFWmonitor* monitor);
Napi::Value glfwGetGammaRamp(Napi::CallbackInfo const& info);

// GLFWAPI void glfwSetGammaRamp(GLFWmonitor* monitor, const GLFWgammaramp* ramp);
void glfwSetGammaRamp(Napi::CallbackInfo const& info);

// GLFWAPI void glfwDefaultWindowHints(void);
Napi::Value glfwDefaultWindowHints(Napi::CallbackInfo const& info);

// GLFWAPI void glfwWindowHint(int hint, int value);
Napi::Value glfwWindowHint(Napi::CallbackInfo const& info);

// GLFWAPI void glfwWindowHintString(int hint, const char* value);
Napi::Value glfwWindowHintString(Napi::CallbackInfo const& info);

// GLFWAPI GLFWwindow* glfwCreateWindow(int width, int height, const char* title, GLFWmonitor*
// monitor, GLFWwindow* share);
Napi::Value glfwCreateWindow(Napi::CallbackInfo const& info);

#ifdef __linux__
// void glfwReparentWindow(GLFWwindow* window);
Napi::Value glfwReparentWindow(Napi::CallbackInfo const& info);
#endif

// GLFWAPI void glfwDestroyWindow(GLFWwindow* window);
Napi::Value glfwDestroyWindow(Napi::CallbackInfo const& info);

// GLFWAPI int glfwWindowShouldClose(GLFWwindow* window);
Napi::Value glfwWindowShouldClose(Napi::CallbackInfo const& info);

// GLFWAPI void glfwSetWindowShouldClose(GLFWwindow* window, int value);
void glfwSetWindowShouldClose(Napi::CallbackInfo const& info);

// GLFWAPI void glfwSetWindowTitle(GLFWwindow* window, const char* title);
void glfwSetWindowTitle(Napi::CallbackInfo const& info);

// GLFWAPI void glfwSetWindowIcon(GLFWwindow* window, int count, const GLFWimage* images);
void glfwSetWindowIcon(Napi::CallbackInfo const& info);

// GLFWAPI void glfwGetWindowPos(GLFWwindow* window, int* xpos, int* ypos);
Napi::Value glfwGetWindowPos(Napi::CallbackInfo const& info);

// GLFWAPI void glfwSetWindowPos(GLFWwindow* window, int xpos, int ypos);
void glfwSetWindowPos(Napi::CallbackInfo const& info);

// GLFWAPI void glfwGetWindowSize(GLFWwindow* window, int* width, int* height);
Napi::Value glfwGetWindowSize(Napi::CallbackInfo const& info);

// GLFWAPI void glfwSetWindowSizeLimits(GLFWwindow* window, int minwidth, int minheight, int
// maxwidth, int maxheight);
void glfwSetWindowSizeLimits(Napi::CallbackInfo const& info);

// GLFWAPI void glfwSetWindowAspectRatio(GLFWwindow* window, int numer, int denom);
void glfwSetWindowAspectRatio(Napi::CallbackInfo const& info);

// GLFWAPI void glfwSetWindowSize(GLFWwindow* window, int width, int height);
void glfwSetWindowSize(Napi::CallbackInfo const& info);

// GLFWAPI void glfwGetFramebufferSize(GLFWwindow* window, int* width, int* height);
Napi::Value glfwGetFramebufferSize(Napi::CallbackInfo const& info);

// GLFWAPI void glfwGetWindowFrameSize(GLFWwindow* window, int* left, int* top, int* right, int*
// bottom);
Napi::Value glfwGetWindowFrameSize(Napi::CallbackInfo const& info);

// GLFWAPI void glfwGetWindowContentScale(GLFWwindow* window, float* xscale, float* yscale);
Napi::Value glfwGetWindowContentScale(Napi::CallbackInfo const& info);

// GLFWAPI float glfwGetWindowOpacity(GLFWwindow* window);
Napi::Value glfwGetWindowOpacity(Napi::CallbackInfo const& info);

// GLFWAPI void glfwSetWindowOpacity(GLFWwindow* window, float opacity);
void glfwSetWindowOpacity(Napi::CallbackInfo const& info);

// GLFWAPI void glfwIconifyWindow(GLFWwindow* window);
Napi::Value glfwIconifyWindow(Napi::CallbackInfo const& info);

// GLFWAPI void glfwRestoreWindow(GLFWwindow* window);
Napi::Value glfwRestoreWindow(Napi::CallbackInfo const& info);

// GLFWAPI void glfwMaximizeWindow(GLFWwindow* window);
Napi::Value glfwMaximizeWindow(Napi::CallbackInfo const& info);

// GLFWAPI void glfwShowWindow(GLFWwindow* window);
Napi::Value glfwShowWindow(Napi::CallbackInfo const& info);

// GLFWAPI void glfwHideWindow(GLFWwindow* window);
Napi::Value glfwHideWindow(Napi::CallbackInfo const& info);

// GLFWAPI void glfwFocusWindow(GLFWwindow* window);
Napi::Value glfwFocusWindow(Napi::CallbackInfo const& info);

// GLFWAPI void glfwRequestWindowAttention(GLFWwindow* window);
Napi::Value glfwRequestWindowAttention(Napi::CallbackInfo const& info);

// GLFWAPI GLFWmonitor* glfwGetWindowMonitor(GLFWwindow* window);
Napi::Value glfwGetWindowMonitor(Napi::CallbackInfo const& info);

// GLFWAPI void glfwSetWindowMonitor(GLFWwindow* window, GLFWmonitor* monitor, int xpos, int ypos,
// int width, int height, int refreshRate);
void glfwSetWindowMonitor(Napi::CallbackInfo const& info);

// GLFWAPI int glfwGetWindowAttrib(GLFWwindow* window, int attrib);
Napi::Value glfwGetWindowAttrib(Napi::CallbackInfo const& info);

// GLFWAPI void glfwSetWindowAttrib(GLFWwindow* window, int attrib, int value);
void glfwSetWindowAttrib(Napi::CallbackInfo const& info);

// GLFWAPI void glfwSetWindowUserPointer(GLFWwindow* window, void* pointer);
// void glfwSetWindowUserPointer(Napi::CallbackInfo const& info);

// GLFWAPI void* glfwGetWindowUserPointer(GLFWwindow* window);
// Napi::Value glfwGetWindowUserPointer(Napi::CallbackInfo const& info);

// GLFWAPI GLFWwindowposfun glfwSetWindowPosCallback(GLFWwindow* window, GLFWwindowposfun callback);
void glfwSetWindowPosCallback(Napi::CallbackInfo const& info);

// GLFWAPI GLFWwindowsizefun glfwSetWindowSizeCallback(GLFWwindow* window, GLFWwindowsizefun
// callback);
void glfwSetWindowSizeCallback(Napi::CallbackInfo const& info);

// GLFWAPI GLFWwindowclosefun glfwSetWindowCloseCallback(GLFWwindow* window, GLFWwindowclosefun
// callback);
void glfwSetWindowCloseCallback(Napi::CallbackInfo const& info);

// GLFWAPI GLFWwindowrefreshfun glfwSetWindowRefreshCallback(GLFWwindow* window,
// GLFWwindowrefreshfun callback);
void glfwSetWindowRefreshCallback(Napi::CallbackInfo const& info);

// GLFWAPI GLFWwindowfocusfun glfwSetWindowFocusCallback(GLFWwindow* window, GLFWwindowfocusfun
// callback);
void glfwSetWindowFocusCallback(Napi::CallbackInfo const& info);

// GLFWAPI GLFWwindowiconifyfun glfwSetWindowIconifyCallback(GLFWwindow* window,
// GLFWwindowiconifyfun callback);
void glfwSetWindowIconifyCallback(Napi::CallbackInfo const& info);

// GLFWAPI GLFWwindowmaximizefun glfwSetWindowMaximizeCallback(GLFWwindow* window,
// GLFWwindowmaximizefun callback);
void glfwSetWindowMaximizeCallback(Napi::CallbackInfo const& info);

// GLFWAPI GLFWframebuffersizefun glfwSetFramebufferSizeCallback(GLFWwindow* window,
// GLFWframebuffersizefun callback);
void glfwSetFramebufferSizeCallback(Napi::CallbackInfo const& info);

// GLFWAPI GLFWwindowcontentscalefun glfwSetWindowContentScaleCallback(GLFWwindow* window,
// GLFWwindowcontentscalefun callback);
void glfwSetWindowContentScaleCallback(Napi::CallbackInfo const& info);

// GLFWAPI void glfwPollEvents(void);
void glfwPollEvents(Napi::CallbackInfo const& info);

// GLFWAPI void glfwWaitEvents(void);
void glfwWaitEvents(Napi::CallbackInfo const& info);

// GLFWAPI void glfwWaitEventsTimeout(double timeout);
void glfwWaitEventsTimeout(Napi::CallbackInfo const& info);

// GLFWAPI void glfwPostEmptyEvent(void);
void glfwPostEmptyEvent(Napi::CallbackInfo const& info);

// GLFWAPI int glfwGetInputMode(GLFWwindow* window, int mode);
Napi::Value glfwGetInputMode(Napi::CallbackInfo const& info);

// GLFWAPI void glfwSetInputMode(GLFWwindow* window, int mode, int value);
void glfwSetInputMode(Napi::CallbackInfo const& info);

// GLFWAPI int glfwRawMouseMotionSupported(void);
Napi::Value glfwRawMouseMotionSupported(Napi::CallbackInfo const& info);

// GLFWAPI const char* glfwGetKeyName(int key, int scancode);
Napi::Value glfwGetKeyName(Napi::CallbackInfo const& info);

// GLFWAPI int glfwGetKeyScancode(int key);
Napi::Value glfwGetKeyScancode(Napi::CallbackInfo const& info);

// GLFWAPI int glfwGetKey(GLFWwindow* window, int key);
Napi::Value glfwGetKey(Napi::CallbackInfo const& info);

// GLFWAPI int glfwGetMouseButton(GLFWwindow* window, int button);
Napi::Value glfwGetMouseButton(Napi::CallbackInfo const& info);

// GLFWAPI void glfwGetCursorPos(GLFWwindow* window, double* xpos, double* ypos);
Napi::Value glfwGetCursorPos(Napi::CallbackInfo const& info);

// GLFWAPI void glfwSetCursorPos(GLFWwindow* window, double xpos, double ypos);
void glfwSetCursorPos(Napi::CallbackInfo const& info);

// GLFWAPI GLFWcursor* glfwCreateCursor(const GLFWimage* image, int xhot, int yhot);
Napi::Value glfwCreateCursor(Napi::CallbackInfo const& info);

// GLFWAPI GLFWcursor* glfwCreateStandardCursor(int shape);
Napi::Value glfwCreateStandardCursor(Napi::CallbackInfo const& info);

// GLFWAPI void glfwDestroyCursor(GLFWcursor* cursor);
void glfwDestroyCursor(Napi::CallbackInfo const& info);

// GLFWAPI void glfwSetCursor(GLFWwindow* window, GLFWcursor* cursor);
void glfwSetCursor(Napi::CallbackInfo const& info);

// GLFWAPI GLFWkeyfun glfwSetKeyCallback(GLFWwindow* window, GLFWkeyfun callback);
void glfwSetKeyCallback(Napi::CallbackInfo const& info);

// GLFWAPI GLFWcharfun glfwSetCharCallback(GLFWwindow* window, GLFWcharfun callback);
void glfwSetCharCallback(Napi::CallbackInfo const& info);

// GLFWAPI GLFWcharmodsfun glfwSetCharModsCallback(GLFWwindow* window, GLFWcharmodsfun callback);
void glfwSetCharModsCallback(Napi::CallbackInfo const& info);

// GLFWAPI GLFWmousebuttonfun glfwSetMouseButtonCallback(GLFWwindow* window, GLFWmousebuttonfun
// callback);
void glfwSetMouseButtonCallback(Napi::CallbackInfo const& info);

// GLFWAPI GLFWcursorposfun glfwSetCursorPosCallback(GLFWwindow* window, GLFWcursorposfun callback);
void glfwSetCursorPosCallback(Napi::CallbackInfo const& info);

// GLFWAPI GLFWcursorenterfun glfwSetCursorEnterCallback(GLFWwindow* window, GLFWcursorenterfun
// callback);
void glfwSetCursorEnterCallback(Napi::CallbackInfo const& info);

// GLFWAPI GLFWscrollfun glfwSetScrollCallback(GLFWwindow* window, GLFWscrollfun callback);
void glfwSetScrollCallback(Napi::CallbackInfo const& info);

// GLFWAPI GLFWdropfun glfwSetDropCallback(GLFWwindow* window, GLFWdropfun callback);
void glfwSetDropCallback(Napi::CallbackInfo const& info);

// GLFWAPI int glfwJoystickPresent(int jid);
Napi::Value glfwJoystickPresent(Napi::CallbackInfo const& info);

// GLFWAPI const float* glfwGetJoystickAxes(int jid, int* count);
Napi::Value glfwGetJoystickAxes(Napi::CallbackInfo const& info);

// GLFWAPI const unsigned char* glfwGetJoystickButtons(int jid, int* count);
Napi::Value glfwGetJoystickButtons(Napi::CallbackInfo const& info);

// GLFWAPI const unsigned char* glfwGetJoystickHats(int jid, int* count);
Napi::Value glfwGetJoystickHats(Napi::CallbackInfo const& info);

// GLFWAPI const char* glfwGetJoystickName(int jid);
Napi::Value glfwGetJoystickName(Napi::CallbackInfo const& info);

// GLFWAPI const char* glfwGetJoystickGUID(int jid);
Napi::Value glfwGetJoystickGUID(Napi::CallbackInfo const& info);

// // GLFWAPI void glfwSetJoystickUserPointer(int jid, void* pointer);
// void glfwSetJoystickUserPointer(Napi::CallbackInfo const& info);

// // GLFWAPI void* glfwGetJoystickUserPointer(int jid);
// Napi::Value glfwGetJoystickUserPointer(Napi::CallbackInfo const& info);

// GLFWAPI int glfwJoystickIsGamepad(int jid);
Napi::Value glfwJoystickIsGamepad(Napi::CallbackInfo const& info);

// GLFWAPI GLFWjoystickfun glfwSetJoystickCallback(GLFWjoystickfun callback);
void glfwSetJoystickCallback(Napi::CallbackInfo const& info);

// GLFWAPI int glfwUpdateGamepadMappings(const char* string);
void glfwUpdateGamepadMappings(Napi::CallbackInfo const& info);

// GLFWAPI const char* glfwGetGamepadName(int jid);
Napi::Value glfwGetGamepadName(Napi::CallbackInfo const& info);

// GLFWAPI int glfwGetGamepadState(int jid, GLFWgamepadstate* state);
Napi::Value glfwGetGamepadState(Napi::CallbackInfo const& info);

// GLFWAPI void glfwSetClipboardString(GLFWwindow* window, const char* string);
void glfwSetClipboardString(Napi::CallbackInfo const& info);

// GLFWAPI const char* glfwGetClipboardString(GLFWwindow* window);
Napi::Value glfwGetClipboardString(Napi::CallbackInfo const& info);

// GLFWAPI double glfwGetTime(void);
Napi::Value glfwGetTime(Napi::CallbackInfo const& info);

// GLFWAPI void glfwSetTime(double time);
void glfwSetTime(Napi::CallbackInfo const& info);

// GLFWAPI uint64_t glfwGetTimerValue(void);
Napi::Value glfwGetTimerValue(Napi::CallbackInfo const& info);

// GLFWAPI uint64_t glfwGetTimerFrequency(void);
Napi::Value glfwGetTimerFrequency(Napi::CallbackInfo const& info);

// GLFWAPI void glfwMakeContextCurrent(GLFWwindow* window);
void glfwMakeContextCurrent(Napi::CallbackInfo const& info);

// GLFWAPI GLFWwindow* glfwGetCurrentContext(void);
Napi::Value glfwGetCurrentContext(Napi::CallbackInfo const& info);

// GLFWAPI void glfwSwapBuffers(GLFWwindow* window);
void glfwSwapBuffers(Napi::CallbackInfo const& info);

// GLFWAPI void glfwSwapInterval(int interval);
void glfwSwapInterval(Napi::CallbackInfo const& info);

// GLFWAPI int glfwExtensionSupported(const char* extension);
Napi::Value glfwExtensionSupported(Napi::CallbackInfo const& info);

// GLFWAPI GLFWglproc glfwGetProcAddress(const char* procname);
Napi::Value glfwGetProcAddress(Napi::CallbackInfo const& info);

// GLFWAPI int glfwVulkanSupported(void);
Napi::Value glfwVulkanSupported(Napi::CallbackInfo const& info);

// GLFWAPI const char** glfwGetRequiredInstanceExtensions(uint32_t* count);
Napi::Value glfwGetRequiredInstanceExtensions(Napi::CallbackInfo const& info);

// TODO:

// #if defined(VK_VERSION_1_0)

// // GLFWAPI GLFWvkproc glfwGetInstanceProcAddress(VkInstance instance, const char* procname);
// Napi::Value glfwGetInstanceProcAddress(Napi::CallbackInfo const& info);

// // GLFWAPI int glfwGetPhysicalDevicePresentationSupport(VkInstance instance, VkPhysicalDevice
// device, uint32_t
// // queuefamily);
// Napi::Value glfwGetPhysicalDevicePresentationSupport(Napi::CallbackInfo const& info);

// // GLFWAPI VkResult glfwCreateWindowSurface(VkInstance instance, GLFWwindow* window, const
// VkAllocationCallbacks*
// // allocator, VkSurfaceKHR* surface);
// Napi::Value glfwCreateWindowSurface(Napi::CallbackInfo const& info);

// #endif

}  // namespace nv
