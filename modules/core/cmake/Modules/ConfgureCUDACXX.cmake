#=============================================================================
# Copyright (c) 2020, NVIDIA CORPORATION.
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

# https://cmake.org/cmake/help/v3.18/module/FetchContent.html
include(FetchContent)

FetchContent_Declare(
    libcudacxx
    GIT_REPOSITORY https://github.com/NVIDIA/libcudacxx.git
    GIT_TAG        1.4.0
    GIT_SHALLOW    true
)

FetchContent_GetProperties(libcudacxx)
if(NOT libcudacxx_POPULATED)
  FetchContent_Populate(libcudacxx)
  # libcudacxx has no CMake targets, so no need to call `add_subdirectory()`.
endif()

set(LIBCUDACXX_INCLUDE_DIR "${libcudacxx_SOURCE_DIR}/include")

message(STATUS "LIBCUDACXX_INCLUDE_DIR: ${LIBCUDACXX_INCLUDE_DIR}")
