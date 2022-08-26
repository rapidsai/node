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
include_guard(GLOBAL)

# If `CMAKE_CUDA_ARCHITECTURES` is not defined, build for all supported architectures. If
# `CMAKE_CUDA_ARCHITECTURES` is set to an empty string (""), build for only the current
# architecture. If `CMAKE_CUDA_ARCHITECTURES` is specified by the user, use user setting.

# This needs to be run before enabling the CUDA language due to the default initialization behavior
# of `CMAKE_CUDA_ARCHITECTURES`, https://gitlab.kitware.com/cmake/cmake/-/issues/21302

set(NODE_RAPIDS_CMAKE_BUILD_FOR_ALL_CUDA_ARCHS FALSE)
set(NODE_RAPIDS_CMAKE_BUILD_FOR_DETECTED_ARCHS FALSE)

if(DEFINED ENV{CUDAARCHS})
    if("$ENV{CUDAARCHS}" STREQUAL "")
        # If CUDAARCHS is <empty_string>, auto-detect current GPU arch
        set(NODE_RAPIDS_CMAKE_BUILD_FOR_DETECTED_ARCHS TRUE)
        message(STATUS "Auto-detecting GPU architecture because the CUDAARCHS environment variable = '")
    elseif("$ENV{CUDAARCHS}" STREQUAL "ALL")
        # If CUDAARCHS is "ALL," build for all supported archs
        set(NODE_RAPIDS_CMAKE_BUILD_FOR_ALL_CUDA_ARCHS TRUE)
        message(STATUS "Building all supported GPU architectures because the CUDAARCHS environment variable = 'ALL'")
    else()
        # Use the current value of the CUDAARCHS env var
        set(CMAKE_CUDA_ARCHITECTURES "$ENV{CUDAARCHS}")
        message(STATUS "Using GPU architectures from CUDAARCHS env var: $ENV{CUDAARCHS}")
    endif()
elseif(DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        # If CMAKE_CUDA_ARCHITECTURES is <empty_string>, auto-detect current GPU arch
        set(NODE_RAPIDS_CMAKE_BUILD_FOR_DETECTED_ARCHS TRUE)
        message(STATUS "Auto-detecting GPU architecture because CMAKE_CUDA_ARCHITECTURES = ''")
    elseif(CMAKE_CUDA_ARCHITECTURES STREQUAL "ALL")
        # If CMAKE_CUDA_ARCHITECTURES is "ALL," build for all supported archs
        set(NODE_RAPIDS_CMAKE_BUILD_FOR_ALL_CUDA_ARCHS TRUE)
        message(STATUS "Building all supported GPU architectures because CMAKE_CUDA_ARCHITECTURES = 'ALL'")
    else()
        # Use the current value of CMAKE_CUDA_ARCHITECTURES
        message(STATUS "Using GPU architectures defined in CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()
else()
    # Fall-back to auto-detecting the current GPU architecture
    set(NODE_RAPIDS_CMAKE_BUILD_FOR_DETECTED_ARCHS TRUE)
    message(STATUS "Auto-detecting GPU architectures because CUDAARCHS env var is not defined, and CMAKE_CUDA_ARCHITECTURES was not specified.")
endif()

# Build the list of supported architectures

set(SUPPORTED_CUDA_ARCHITECTURES "60" "70" "75" "80" "86")

find_package(CUDAToolkit REQUIRED)

# CMake < 3.20 has a bug in FindCUDAToolkit where it won't properly detect the CUDAToolkit version
# when find_package(CUDAToolkit) occurs before enable_language(CUDA)
if(NOT DEFINED CUDAToolkit_VERSION AND CMAKE_CUDA_COMPILER)
    execute_process(COMMAND ${CMAKE_CUDA_COMPILER} "--version" OUTPUT_VARIABLE NVCC_OUT)
    if(NVCC_OUT MATCHES [=[ V([0-9]+)\.([0-9]+)\.([0-9]+)]=])
        set(CUDAToolkit_VERSION_MAJOR "${CMAKE_MATCH_1}")
        set(CUDAToolkit_VERSION_MINOR "${CMAKE_MATCH_2}")
        set(CUDAToolkit_VERSION_PATCH "${CMAKE_MATCH_3}")
        set(CUDAToolkit_VERSION "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
    endif()
    unset(NVCC_OUT)
endif()

if(CUDAToolkit_VERSION_MAJOR EQUAL 11 AND CUDAToolkit_VERSION_MINOR LESS 2)
    list(REMOVE_ITEM SUPPORTED_CUDA_ARCHITECTURES "86")
endif()
if(CUDAToolkit_VERSION_MAJOR LESS 11)
    list(REMOVE_ITEM SUPPORTED_CUDA_ARCHITECTURES "86")
    list(REMOVE_ITEM SUPPORTED_CUDA_ARCHITECTURES "80")
endif()
if(CUDAToolkit_VERSION_MAJOR LESS 10)
    list(REMOVE_ITEM SUPPORTED_CUDA_ARCHITECTURES "75")
endif()
if(CUDAToolkit_VERSION_MAJOR LESS 9)
    list(REMOVE_ITEM SUPPORTED_CUDA_ARCHITECTURES "70")
endif()

if(NODE_RAPIDS_CMAKE_BUILD_FOR_ALL_CUDA_ARCHS)
    set(CMAKE_CUDA_ARCHITECTURES ${SUPPORTED_CUDA_ARCHITECTURES})
    # CMake architecture list entry of "80" means to build compute and sm. What we want is for the
    # newest arch only to build that way while the rest built only for sm.
    list(POP_BACK CMAKE_CUDA_ARCHITECTURES latest_arch)
    list(TRANSFORM CMAKE_CUDA_ARCHITECTURES APPEND "-real")
    list(APPEND CMAKE_CUDA_ARCHITECTURES ${latest_arch})
elseif(NODE_RAPIDS_CMAKE_BUILD_FOR_DETECTED_ARCHS)
    # Auto-detect available GPU compute architectures
    execute_process(COMMAND node -p
                    "require('@rapidsai/core').cmake_modules_path"
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                    OUTPUT_VARIABLE NODE_RAPIDS_CMAKE_MODULES_PATH
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    include(${NODE_RAPIDS_CMAKE_MODULES_PATH}/EvalGpuArchs.cmake)
    evaluate_gpu_archs(CMAKE_CUDA_ARCHITECTURES)
    list(TRANSFORM CMAKE_CUDA_ARCHITECTURES APPEND "-real")
endif()

message(STATUS "BUILD_FOR_DETECTED_ARCHS: ${NODE_RAPIDS_CMAKE_BUILD_FOR_DETECTED_ARCHS}")
message(STATUS "BUILD_FOR_ALL_CUDA_ARCHS: ${NODE_RAPIDS_CMAKE_BUILD_FOR_ALL_CUDA_ARCHS}")
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

# Override the cached version from enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" CACHE STRING "" FORCE)

set(BLA_STATIC ON)
set(CUDA_STATIC_RUNTIME ON)
set(CUDA_USE_STATIC_CUDA_RUNTIME ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY STATIC)

# Enable the CUDA language
enable_language(CUDA)

list(APPEND NODE_RAPIDS_CMAKE_CUDA_FLAGS -Werror=cross-execution-space-call)
list(APPEND NODE_RAPIDS_CMAKE_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)
list(APPEND NODE_RAPIDS_CMAKE_CUDA_FLAGS -Xcompiler=-Wall,-Werror,-Wno-error=deprecated-declarations)
