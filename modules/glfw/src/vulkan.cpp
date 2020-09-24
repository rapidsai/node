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

#include <nv_node/utilities/cpp_to_napi.hpp>

namespace nv {

// GLFWAPI int glfwVulkanSupported(void);
Napi::Value glfwVulkanSupported(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(static_cast<bool>(GLFWAPI::glfwVulkanSupported()));
}

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

// #endif /*VK_VERSION_1_0*/

}  // namespace nv
