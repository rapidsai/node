#=============================================================================
# Copyright (c) 2020-2026, NVIDIA CORPORATION.
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

if(POLICY CMP0099)
  # Link properties are transitive over private dependencies of static libraries
  # https://cmake.org/cmake/help/latest/policy/CMP0099.html#policy:CMP0099
  cmake_policy(SET CMP0099 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0099 NEW)
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

if(POLICY CMP0169)
  # Allow calling FetchContent_Populate with declared details
  # https://cmake.org/cmake/help/latest/policy/CMP0169.html
  cmake_policy(SET CMP0169 OLD)
  set(CMAKE_POLICY_DEFAULT_CMP0169 OLD)
endif()

if(POLICY CMP0177)
  # install() DESTINATION paths are normalized
  # https://cmake.org/cmake/help/latest/policy/CMP0177.html
  cmake_policy(SET CMP0177 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0177 NEW)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
unset(CMAKE_LIBRARY_OUTPUT_DIRECTORY)
unset(CMAKE_LIBRARY_OUTPUT_DIRECTORY CACHE)

option(NODE_RAPIDS_USE_SCCACHE "Enable caching compilation results with sccache" ON)

# Don't use sccache-dist for CMake's compiler tests
set(ENV{SCCACHE_NO_DIST_COMPILE} "1")
