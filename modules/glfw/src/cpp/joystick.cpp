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

// GLFWAPI int glfwJoystickPresent(int jid);
Napi::Value glfwJoystickPresent(Napi::CallbackInfo const& info) {
  return ToNapi(info.Env())(GLFWAPI::glfwJoystickPresent(FromJS(info[0])));
}

// GLFWAPI const float* glfwGetJoystickAxes(int jid, int* count);
Napi::Value glfwGetJoystickAxes(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  int32_t count{};
  const float* axes = GLFWAPI::glfwGetJoystickAxes(FromJS(info[0]), &count);
  return ToNapi(env)(std::vector<float>{axes, axes + count});
}

// GLFWAPI const unsigned char* glfwGetJoystickButtons(int jid, int* count);
Napi::Value glfwGetJoystickButtons(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  int32_t count{};
  const uint8_t* btns = GLFWAPI::glfwGetJoystickButtons(FromJS(info[0]), &count);
  return ToNapi(env)(std::vector<uint8_t>{btns, btns + count});
}

// GLFWAPI const unsigned char* glfwGetJoystickHats(int jid, int* count);
Napi::Value glfwGetJoystickHats(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  int32_t count{};
  const uint8_t* hats = GLFWAPI::glfwGetJoystickHats(FromJS(info[0]), &count);
  return ToNapi(env)(std::vector<uint8_t>{hats, hats + count});
}

// GLFWAPI const char* glfwGetJoystickName(int jid);
Napi::Value glfwGetJoystickName(Napi::CallbackInfo const& info) {
  return ToNapi(info.Env())(GLFWAPI::glfwGetJoystickName(FromJS(info[0])));
}

// GLFWAPI const char* glfwGetJoystickGUID(int jid);
Napi::Value glfwGetJoystickGUID(Napi::CallbackInfo const& info) {
  return ToNapi(info.Env())(GLFWAPI::glfwGetJoystickGUID(FromJS(info[0])));
}

// GLFWAPI void glfwSetJoystickUserPointer(int jid, void* pointer);
// Napi::Value glfwSetJoystickUserPointer(Napi::CallbackInfo const& info);

// GLFWAPI void* glfwGetJoystickUserPointer(int jid);
// Napi::Value glfwGetJoystickUserPointer(Napi::CallbackInfo const& info);

// GLFWAPI int glfwJoystickIsGamepad(int jid);
Napi::Value glfwJoystickIsGamepad(Napi::CallbackInfo const& info) {
  return ToNapi(info.Env())(static_cast<bool>(GLFWAPI::glfwJoystickIsGamepad(FromJS(info[0]))));
}

// GLFWAPI int glfwUpdateGamepadMappings(const char* string);
Napi::Value glfwUpdateGamepadMappings(Napi::CallbackInfo const& info) {
  auto env             = info.Env();
  std::string mappings = FromJS(info[0]);
  GLFW_TRY(env, GLFWAPI::glfwUpdateGamepadMappings(mappings.data()));
  return env.Undefined();
}

// GLFWAPI const char* glfwGetGamepadName(int jid);
Napi::Value glfwGetGamepadName(Napi::CallbackInfo const& info) {
  return ToNapi(info.Env())(GLFWAPI::glfwGetGamepadName(FromJS(info[0])));
}

// GLFWAPI int glfwGetGamepadState(int jid, GLFWgamepadstate* state);
Napi::Value glfwGetGamepadState(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GLFWAPI::GLFWgamepadstate state{};
  GLFW_TRY(env, GLFWAPI::glfwGetGamepadState(FromJS(info[0]), &state));
  return ToNapi(env)(state);
}

}  // namespace node_glfw
