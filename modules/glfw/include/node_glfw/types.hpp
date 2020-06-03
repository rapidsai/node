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

#pragma once

#include <cstdlib>
#include <visit_struct/visit_struct.hpp>
#include "glfw.hpp"

VISITABLE_STRUCT(GLFWAPI::GLFWvidmode, width, height, redBits, greenBits, blueBits, refreshRate);

VISITABLE_STRUCT(GLFWAPI::GLFWgammaramp, red, green, blue, size);

VISITABLE_STRUCT(GLFWAPI::GLFWimage, width, height, pixels);

VISITABLE_STRUCT(GLFWAPI::GLFWgamepadstate, buttons, axes);

static_assert(visit_struct::traits::is_visitable<GLFWAPI::GLFWvidmode>::value, "");

static_assert(visit_struct::traits::is_visitable<GLFWAPI::GLFWgammaramp>::value, "");
