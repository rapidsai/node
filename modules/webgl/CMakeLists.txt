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
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(node_webgl VERSION 0.0.1 LANGUAGES C CXX)

# Set OpenGL_GL_PREFERENCE to "GLVND"
cmake_policy(SET CMP0072 NEW)

set(NODE_WEBGL_STATIC_LINK OFF)
set(REQUIRED_GLEW_VERSION "2.1.0")
set(REQUIRED_GLFW_VERSION "3.3.2")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(GLEW_USE_STATIC_LIBS ${NODE_WEBGL_STATIC_LINK})
set(GLFW_USE_STATIC_LIBS ${NODE_WEBGL_STATIC_LINK})
set(GLEW_USE_SHARED_LIBS ON)
set(GLFW_USE_SHARED_LIBS ON)

if(NODE_WEBGL_STATIC_LINK)
    set(GLEW_USE_STATIC_LIBS ON)
    set(GLFW_USE_STATIC_LIBS ON)
    set(GLEW_USE_SHARED_LIBS OFF)
    set(GLFW_USE_SHARED_LIBS OFF)
endif(NODE_WEBGL_STATIC_LINK)

if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

set(PARALLEL_BUILD "")
if($ENV{PARALLEL_LEVEL})
    set(NUM_JOBS $ENV{PARALLEL_LEVEL})
    set(PARALLEL_BUILD "-j${NUM_JOBS}")
endif($ENV{PARALLEL_LEVEL})

if(${NUM_JOBS})
    if(${NUM_JOBS} EQUAL 1)
        message(STATUS "Enabling Sequential CMake build")
    elseif(${NUM_JOBS} GREATER 1)
        message(STATUS "Enabling Parallel CMake build with ${NUM_JOBS} jobs")
    endif(${NUM_JOBS} EQUAL 1)
else()
    message(STATUS "Enabling Parallel CMake build with all threads")
endif(${NUM_JOBS})

###################################################################################################
# - cmake modules ---------------------------------------------------------------------------------

execute_process(COMMAND npx nvidia-cmake-modules-path
                OUTPUT_VARIABLE NVIDIA_CMAKE_MODULES_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${NVIDIA_CMAKE_MODULES_PATH}")

include(ConfigureNapi)
include(ConfigureDisplayServer)
include(ConfigureOpenGL)
include(ConfigureOpenGLEW)
include(ConfigureOpenGLFW)

include(FeatureSummary)
include(CheckIncludeFiles)
include(CheckLibraryExists)

###################################################################################################
# - compiler options ------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_WIN32")
elseif(LINUX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__linux__")
elseif(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__APPLE__")
endif()

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${GLEW_INCLUDE_DIRS}"
                    "${GLFW_INCLUDE_DIRS}"
                    "${OPENGL_INCLUDE_DIR}"
                    "${DISPLAY_SERVER_INCLUDE_DIRS}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/include")

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

file(GLOB_RECURSE NODE_WEBGL_SRC_FILES "src/cpp/*.cpp")

add_library(${PROJECT_NAME} SHARED ${NODE_WEBGL_SRC_FILES} ${CMAKE_JS_SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")
target_link_libraries(${PROJECT_NAME}
                      ${CMAKE_JS_LIB}
                      ${GLEW_LIBRARY}
                      ${GLFW_LIBRARY}
                      OpenGL::GL)
