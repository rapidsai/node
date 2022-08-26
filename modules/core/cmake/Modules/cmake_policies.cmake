#=============================================================================
# Copyright (c) 2020-2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================
include_guard(GLOBAL)

if(POLICY CMP0072)
  # Set OpenGL_GL_PREFERENCE to "GLVND"
  # https://cmake.org/cmake/help/latest/policy/CMP0072.html#policy:CMP0072
  cmake_policy(SET CMP0072 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0072 NEW)
endif()

if(POLICY CMP0102)
  # empty cache variable sets
  # https://cmake.org/cmake/help/latest/policy/CMP0102.html#policy:CMP0102
  cmake_policy(SET CMP0102 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0102 NEW)
endif()

if(POLICY CMP0124)
  # unset loop variables
  # https://cmake.org/cmake/help/latest/policy/CMP0124.html#policy:CMP0124
  cmake_policy(SET CMP0124 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0124 NEW)
endif()

if(POLICY CMP0126)
  # make set(CACHE) command not remove normal variable of the same name from the current scope
  # https://cmake.org/cmake/help/latest/policy/CMP0126.html
  cmake_policy(SET CMP0126 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0126 NEW)
endif()
