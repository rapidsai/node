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

namespace nv {

// GLFWAPI void glfwPollEvents(void);
void glfwPollEvents(Napi::CallbackInfo const& info) {
  GLFW_TRY(info.Env(), GLFWAPI::glfwPollEvents());
}

// GLFWAPI void glfwWaitEvents(void);
void glfwWaitEvents(Napi::CallbackInfo const& info) {
  GLFW_TRY(info.Env(), GLFWAPI::glfwWaitEvents());
}

// GLFWAPI void glfwWaitEventsTimeout(double timeout);
void glfwWaitEventsTimeout(Napi::CallbackInfo const& info) {
  GLFW_TRY(info.Env(), GLFWAPI::glfwWaitEventsTimeout(info[0].ToNumber()));
}

// GLFWAPI void glfwPostEmptyEvent(void);
void glfwPostEmptyEvent(Napi::CallbackInfo const& info) {
  GLFW_TRY(info.Env(), GLFWAPI::glfwPostEmptyEvent());
}

}  // namespace nv
