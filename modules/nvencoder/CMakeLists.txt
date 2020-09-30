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

project(node_nvencoder VERSION 0.0.1 LANGUAGES C CXX)

option(NVIDIA_USE_CCACHE "Enable caching compilation results with ccache" ON)
option(NODE_NVENCODER_STATIC_LINK "Statically link GLEW library" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(VIDEO_CODEC_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

set(GLEW_VERSION "2.1.0")

set(GLEW_USE_SHARED_LIBS OFF)
set(GLEW_USE_STATIC_LIBS ${NODE_NVENCODER_STATIC_LINK})

if(NOT NODE_NVENCODER_STATIC_LINK)
    set(GLEW_USE_SHARED_LIBS ON)
endif(NOT NODE_NVENCODER_STATIC_LINK)

if(DEFINED ENV{VIDEO_CODEC_SDK_DIR})
    set(VIDEO_CODEC_SDK_DIR $ENV{VIDEO_CODEC_SDK_DIR})
endif(DEFINED ENV{VIDEO_CODEC_SDK_DIR})

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
include(ConfigureCUDA)
include(ConfigureOpenGL)
include(ConfigureOpenGLEW)

find_path(VIDEO_CODEC_SDK_INCLUDE_DIR NO_DEFAULT_PATH REQUIRED
          NAMES nvEncodeAPI.h
          HINTS "${VIDEO_CODEC_SDK_DIR}"
          PATH_SUFFIXES "include" "Interface"
          DOC "The nvEncodeAPI.h parent dir")

find_library(NVENCODEAPI_LIB
    NAMES nvidia-encode
          libnvidia-encode
          libnvidia-encode.so
          libnvidia-encode.so.1)

message(STATUS "NVENCODEAPI_LIB: ${NVENCODEAPI_LIB}")

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${CUDA_INCLUDE_DIRS}"
                    "${RAPIDS_CORE_INCLUDE_DIR}"
                    "${GLEW_INCLUDE_DIR}"
                    "${OPENGL_INCLUDE_DIR}"
                    "${VIDEO_CODEC_SDK_INCLUDE_DIR}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src"
                    "${CMAKE_CURRENT_SOURCE_DIR}/include")

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CUDA_LIBRARIES}")

file(GLOB_RECURSE NODE_NVENCODER_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(${PROJECT_NAME} SHARED ${NODE_NVENCODER_SRC_FILES} ${CMAKE_JS_SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")
target_link_libraries(${PROJECT_NAME}
                      ${CMAKE_JS_LIB}
                      ${GLEW_LIBRARY}
                      ${NVENCODEAPI_LIB}
                      cuda cudart nppicc nppig
                      OpenGL::EGL
                      OpenGL::OpenGL)
