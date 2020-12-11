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
cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(node_webgl VERSION 0.0.1 LANGUAGES C CXX)

option(NVIDIA_USE_CCACHE "Enable caching compilation results with ccache" ON)
option(NODE_WEBGL_STATIC_LINK "Statically link GLEW and GLFW libraries" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set OpenGL_GL_PREFERENCE to "GLVND"
cmake_policy(SET CMP0072 NEW)

set(GLEW_VERSION "2.1.0")
set(GLFW_VERSION "3.3.2")

set(GLEW_USE_SHARED_LIBS OFF)
set(GLFW_USE_SHARED_LIBS OFF)
set(GLEW_USE_STATIC_LIBS ${NODE_WEBGL_STATIC_LINK})
set(GLFW_USE_STATIC_LIBS ${NODE_WEBGL_STATIC_LINK})

if(NOT NODE_WEBGL_STATIC_LINK)
    set(GLEW_USE_SHARED_LIBS ON)
    set(GLFW_USE_SHARED_LIBS ON)
endif(NOT NODE_WEBGL_STATIC_LINK)

###################################################################################################
# - cmake modules ---------------------------------------------------------------------------------

execute_process(COMMAND node -p
                "require('@nvidia/rapids-core').cmake_modules_path"
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE NVIDIA_CMAKE_MODULES_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${NVIDIA_CMAKE_MODULES_PATH}")

include(ConfigureCXX)
include(ConfigureNapi)
include(ConfigureDisplayServer)
include(ConfigureOpenGL)
include(ConfigureOpenGLEW)
include(ConfigureOpenGLFW)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories(${RAPIDS_CORE_INCLUDE_DIR}
                    ${GLEW_INCLUDE_DIR}
                    ${GLFW_INCLUDE_DIR}
                    ${OPENGL_INCLUDE_DIR}
                    ${OPENGL_EGL_INCLUDE_DIRS}
                    ${DISPLAY_SERVER_INCLUDE_DIRS}
                    ${NAPI_INCLUDE_DIRS})

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

file(GLOB_RECURSE NODE_WEBGL_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(${PROJECT_NAME} SHARED ${NODE_WEBGL_SRC_FILES} ${CMAKE_JS_SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")
target_link_libraries(${PROJECT_NAME}
                      ${CMAKE_JS_LIB}
                      ${GLEW_LIBRARY}
                      ${GLFW_LIBRARY}
                      OpenGL::EGL
                      OpenGL::OpenGL)
