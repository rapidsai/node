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
#include "napi.h"

#include <nv_node/utilities/args.hpp>
#include <nv_node/utilities/cpp_to_napi.hpp>

namespace nv {

// GLFWAPI int glfwJoystickPresent(int jid);
Napi::Value glfwJoystickPresent(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  auto present = GLFWAPI::glfwJoystickPresent(args[0]);
  return CPPToNapi(info)(static_cast<bool>(present));
}

// GLFWAPI const float* glfwGetJoystickAxes(int jid, int* count);
Napi::Value glfwGetJoystickAxes(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  int32_t count{};
  float const* axes = GLFWAPI::glfwGetJoystickAxes(args[0], &count);
  return CPPToNapi(info)(std::vector<float>{axes, axes + count});
}

// GLFWAPI const unsigned char* glfwGetJoystickButtons(int jid, int* count);
Napi::Value glfwGetJoystickButtons(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  int32_t count{};
  uint8_t const* btns = GLFWAPI::glfwGetJoystickButtons(args[0], &count);
  return CPPToNapi(info)(std::vector<uint8_t>{btns, btns + count});
}

// GLFWAPI const unsigned char* glfwGetJoystickHats(int jid, int* count);
Napi::Value glfwGetJoystickHats(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  int32_t count{};
  uint8_t const* hats = GLFWAPI::glfwGetJoystickHats(args[0], &count);
  return CPPToNapi(info)(std::vector<uint8_t>{hats, hats + count});
}

// GLFWAPI const char* glfwGetJoystickName(int jid);
Napi::Value glfwGetJoystickName(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  auto name = GLFWAPI::glfwGetJoystickName(args[0]);
  return CPPToNapi(info)(std::string{name == nullptr ? name : ""});
}

// GLFWAPI const char* glfwGetJoystickGUID(int jid);
Napi::Value glfwGetJoystickGUID(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  auto guid = GLFWAPI::glfwGetJoystickGUID(args[0]);
  return CPPToNapi(info)(std::string{guid == nullptr ? guid : ""});
}

// GLFWAPI void glfwSetJoystickUserPointer(int jid, void* pointer);
// Napi::Value glfwSetJoystickUserPointer(Napi::CallbackInfo const& info);

// GLFWAPI void* glfwGetJoystickUserPointer(int jid);
// Napi::Value glfwGetJoystickUserPointer(Napi::CallbackInfo const& info);

// GLFWAPI int glfwJoystickIsGamepad(int jid);
Napi::Value glfwJoystickIsGamepad(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  auto is_gamepad = GLFWAPI::glfwJoystickIsGamepad(args[0]);
  return CPPToNapi(info)(static_cast<bool>(is_gamepad));
}

// GLFWAPI int glfwUpdateGamepadMappings(const char* string);
void glfwUpdateGamepadMappings(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  std::string mappings = args[0];
  GLFW_TRY(env, GLFWAPI::glfwUpdateGamepadMappings(mappings.data()));
}

// GLFWAPI const char* glfwGetGamepadName(int jid);
Napi::Value glfwGetGamepadName(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  auto name = GLFWAPI::glfwGetGamepadName(args[0]);
  return CPPToNapi(info)(std::string{name == nullptr ? name : ""});
}

// GLFWAPI int glfwGetGamepadState(int jid, GLFWgamepadstate* state);
Napi::Value glfwGetGamepadState(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  GLFWAPI::GLFWgamepadstate state{};
  GLFW_TRY(env, GLFWAPI::glfwGetGamepadState(args[0], &state));

  std::vector<float> axes;
  axes.reserve(sizeof(state.axes));
  axes.insert(axes.begin(), state.axes, state.axes + 6);

  std::vector<unsigned char> buttons;
  buttons.reserve(sizeof(state.buttons));
  buttons.insert(buttons.begin(), state.buttons, state.buttons + 15);

  auto js_state = Napi::Object::New(env);
  js_state.Set("axes", CPPToNapi(env)(axes));
  js_state.Set("buttons", CPPToNapi(env)(buttons));
  return js_state;
}

}  // namespace nv
