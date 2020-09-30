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

# include(FindOpenGL REQUIRED)

find_package(OpenGL REQUIRED EGL OpenGL)

if(NOT OPENGL_FOUND)
    message(FATAL_ERROR "OpenGL not found")
endif()

message(STATUS "OpenGL libraries: " ${OPENGL_LIBRARIES})
message(STATUS "OpenGL includes: " ${OPENGL_INCLUDE_DIR})

message(STATUS "OPENGL_egl_LIBRARY: " ${OPENGL_egl_LIBRARY})
message(STATUS "OPENGL_EGL_INCLUDE_DIRS: " ${OPENGL_EGL_INCLUDE_DIRS})
