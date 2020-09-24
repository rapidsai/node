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

namespace nv {

// GLFWAPI void glfwPollEvents(void);
Napi::Value glfwPollEvents(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GLFW_TRY(env, GLFWAPI::glfwPollEvents());
  return env.Undefined();
}

// GLFWAPI void glfwWaitEvents(void);
Napi::Value glfwWaitEvents(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GLFW_TRY(env, GLFWAPI::glfwWaitEvents());
  return env.Undefined();
}

// GLFWAPI void glfwWaitEventsTimeout(double timeout);
Napi::Value glfwWaitEventsTimeout(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  GLFW_TRY(env, GLFWAPI::glfwWaitEventsTimeout(args[0]));
  return env.Undefined();
}

// GLFWAPI void glfwPostEmptyEvent(void);
Napi::Value glfwPostEmptyEvent(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  GLFW_TRY(env, GLFWAPI::glfwPostEmptyEvent());
  return env.Undefined();
}

}  // namespace nv
