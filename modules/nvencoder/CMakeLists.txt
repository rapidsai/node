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

project(node_nvencoder VERSION 0.0.1 LANGUAGES C CXX)

set(VIDEO_CODEC_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/sdk/Video_Codec_SDK_10.0.26")
set(NODE_NVENCODER_STATIC_LINK OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(REQUIRED_GLEW_VERSION "2.1.0")
set(GLEW_USE_STATIC_LIBS ${NODE_NVENCODER_STATIC_LINK})
set(GLEW_USE_SHARED_LIBS ON)

if(NODE_NVENCODER_STATIC_LINK)
    set(GLEW_USE_STATIC_LIBS ON)
    set(GLEW_USE_SHARED_LIBS OFF)
endif(NODE_NVENCODER_STATIC_LINK)

if(DEFINED ENV{VIDEO_CODEC_SDK_DIR})
    set(VIDEO_CODEC_SDK_DIR $ENV{VIDEO_CODEC_SDK_DIR})
endif(DEFINED ENV{VIDEO_CODEC_SDK_DIR})

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
include(ConfigureCUDA)
include(ConfigureOpenGL)
include(ConfigureOpenGLEW)

include(FeatureSummary)
include(CheckIncludeFiles)
include(CheckLibraryExists)

find_path(VIDEO_CODEC_SDK_INCLUDE_DIR nvEncodeAPI.h "${VIDEO_CODEC_SDK_DIR}/include")

find_path(VIDEO_CODEC_SDK_INCLUDE_DIR NO_DEFAULT_PATH REQUIRED
          NAMES nvEncodeAPI.h
          HINTS "${VIDEO_CODEC_SDK_DIR}"
          PATH_SUFFIXES "include" "Interface"
          DOC "The nvEncodeAPI.h parent dir")

find_library(CUVID_LIB nvcuvid)
find_library(NVENCODEAPI_LIB nvidia-encode)

message(STATUS "CUVID_LIB: ${CUVID_LIB}")
message(STATUS "NVENCODEAPI_LIB: ${NVENCODEAPI_LIB}")

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

include_directories("${CUDA_INCLUDE_DIRS}"
                    "${GLEW_INCLUDE_DIRS}"
                    "${OPENGL_INCLUDE_DIR}"
                    "${VIDEO_CODEC_SDK_INCLUDE_DIR}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/include"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp")

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CUDA_LIBRARIES}")

file(GLOB_RECURSE NODE_NVENCODER_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/*.cpp")

add_library(${PROJECT_NAME} SHARED ${NODE_NVENCODER_SRC_FILES} ${CMAKE_JS_SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")
target_link_libraries(${PROJECT_NAME}
                      ${CMAKE_JS_LIB} cuda cudart ${GLEW_LIBRARY} OpenGL::GL ${NVENCODEAPI_LIB} ${CUVID_LIB})
