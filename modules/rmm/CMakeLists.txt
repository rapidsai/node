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

project(node_rmm VERSION 0.0.1 LANGUAGES C CXX)

option(DISABLE_DEPRECATION_WARNINGS "Disable warnings generated from deprecated declarations." ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(REQUIRED_RMM_VERSION "0.15")

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

include(ConfigureCUDA)
include(ConfigureNapi)
include(ConfigureRMM)

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

if(DISABLE_DEPRECATION_WARNINGS)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-deprecated-declarations")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
endif(DISABLE_DEPRECATION_WARNINGS)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${RMM_INCLUDE_DIRS}"
                    "${CUDA_INCLUDE_DIRS}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/include")

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CUDA_LIBRARIES}")

file(GLOB_RECURSE NODE_CUDA_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/*.cpp")

add_library(${PROJECT_NAME} SHARED ${NODE_CUDA_SRC_FILES} ${CMAKE_JS_SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")
target_link_libraries(${PROJECT_NAME}
                      ${CMAKE_JS_LIB} cuda cudart nvrtc)
