#=============================================================================
# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
cmake_minimum_required(VERSION 3.18...3.20 FATAL_ERROR)

project(node_glfw VERSION 0.0.1 LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
unset(CMAKE_LIBRARY_OUTPUT_DIRECTORY)
unset(CMAKE_LIBRARY_OUTPUT_DIRECTORY CACHE)

option(NODE_RAPIDS_USE_CCACHE "Enable caching compilation results with ccache" ON)
option(NODE_GLFW_STATIC_LINK "Statically link GLEW and GLFW libraries" OFF)

# Set OpenGL_GL_PREFERENCE to "GLVND"
cmake_policy(SET CMP0072 NEW)

set(GLEW_VERSION "2.1.0")
set(GLFW_VERSION "3.3.2")

set(GLEW_USE_SHARED_LIBS OFF)
set(GLFW_USE_SHARED_LIBS OFF)
set(GLEW_USE_STATIC_LIBS ${NODE_GLFW_STATIC_LINK})
set(GLFW_USE_STATIC_LIBS ${NODE_GLFW_STATIC_LINK})

if(NOT NODE_GLFW_STATIC_LINK)
    set(GLEW_USE_SHARED_LIBS ON)
    set(GLFW_USE_SHARED_LIBS ON)
endif(NOT NODE_GLFW_STATIC_LINK)

###################################################################################################
# - cmake modules ---------------------------------------------------------------------------------

execute_process(COMMAND node -p
                "require('@rapidsai/core').cmake_modules_path"
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE NODE_RAPIDS_CMAKE_MODULES_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE)

list(APPEND CMAKE_MODULE_PATH "${NODE_RAPIDS_CMAKE_MODULES_PATH}")

include(ConfigureCXX)
include(ConfigureNapi)
include(ConfigureDisplayServer)
include(ConfigureOpenGL)
include(ConfigureOpenGLEW)
include(ConfigureOpenGLFW)

###################################################################################################
# - node_glfw target ------------------------------------------------------------------------------

file(GLOB_RECURSE NODE_GLFW_SRC_FILES "src/*.cpp")

add_library(${PROJECT_NAME} SHARED ${NODE_GLFW_SRC_FILES} ${CMAKE_JS_SRC})

set_target_properties(${PROJECT_NAME}
    PROPERTIES PREFIX                              ""
               SUFFIX                              ".node"
               BUILD_RPATH                         "\$ORIGIN"
               INSTALL_RPATH                       "\$ORIGIN"
               CXX_STANDARD                        17
               CXX_STANDARD_REQUIRED               ON
               CUDA_STANDARD                       17
               CUDA_STANDARD_REQUIRED              ON
               NO_SYSTEM_FROM_IMPORTED             ON
               POSITION_INDEPENDENT_CODE           ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_compile_options(${PROJECT_NAME}
            PRIVATE "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:C>:${NODE_RAPIDS_CMAKE_C_FLAGS}>>"
                    "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:${NODE_RAPIDS_CMAKE_CXX_FLAGS}>>"
)

target_include_directories(${PROJECT_NAME}
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
           "$<BUILD_INTERFACE:${RAPIDS_CORE_INCLUDE_DIR}>"
           "$<BUILD_INTERFACE:${NAPI_INCLUDE_DIRS}>"
           "$<BUILD_INTERFACE:${DISPLAY_SERVER_INCLUDE_DIRS}>"
)

target_link_libraries(${PROJECT_NAME}
                      ${CMAKE_JS_LIB}
                      ${GLEW_LIBRARY}
                      glfw
                      OpenGL::EGL
                      OpenGL::OpenGL)
