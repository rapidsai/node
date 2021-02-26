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

# If `CMAKE_CUDA_ARCHITECTURES` is not defined, build for all supported architectures. If
# `CMAKE_CUDA_ARCHITECTURES` is set to an empty string (""), build for only the current
# architecture. If `CMAKE_CUDA_ARCHITECTURES` is specified by the user, use user setting.

# This needs to be run before enabling the CUDA language due to the default initialization behavior
# of `CMAKE_CUDA_ARCHITECTURES`, https://gitlab.kitware.com/cmake/cmake/-/issues/21302

set(NVIDIA_CMAKE_BUILD_FOR_ALL_CUDA_ARCHS FALSE)
set(NVIDIA_CMAKE_BUILD_FOR_DETECTED_ARCHS FALSE)

if(NOT "$ENV{CUDAARCHS}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "$ENV{CUDAARCHS}")
elseif(CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    unset(CMAKE_CUDA_ARCHITECTURES CACHE)
    set(NVIDIA_CMAKE_BUILD_FOR_DETECTED_ARCHS TRUE)
elseif(NOT DEFINED ENV{CUDAARCHS})
    set(NVIDIA_CMAKE_BUILD_FOR_DETECTED_ARCHS TRUE)
elseif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(NVIDIA_CMAKE_BUILD_FOR_ALL_CUDA_ARCHS TRUE)
endif()

# Enable the CUDA language
enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)

if(CMAKE_CUDA_COMPILER_VERSION)
    # Compute the version. from  CMAKE_CUDA_COMPILER_VERSION
    string(REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\1" CUDA_VERSION_MAJOR ${CMAKE_CUDA_COMPILER_VERSION})
    string(REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\2" CUDA_VERSION_MINOR ${CMAKE_CUDA_COMPILER_VERSION})
    set(CUDA_VERSION "${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}")
endif()

string(APPEND CMAKE_CUDA_FLAGS " -Werror=cross-execution-space-call")
string(APPEND CMAKE_CUDA_FLAGS " --expt-extended-lambda --expt-relaxed-constexpr")
string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler=-Wall,-Werror,-Wno-error=deprecated-declarations")

# Build the list of supported architectures

set(SUPPORTED_CUDA_ARCHITECTURES "60" "62" "70" "72" "75" "80")

# Check for embedded vs workstation architectures
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    # This is being built for Linux4Tegra or SBSA ARM64
    list(REMOVE_ITEM SUPPORTED_CUDA_ARCHITECTURES "60" "70")
else()
    # This is being built for an x86 or x86_64 architecture
    list(REMOVE_ITEM SUPPORTED_CUDA_ARCHITECTURES "62" "72")
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11)
    list(REMOVE_ITEM SUPPORTED_CUDA_ARCHITECTURES "80")
endif()
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 10)
    list(REMOVE_ITEM SUPPORTED_CUDA_ARCHITECTURES "75")
endif()
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 9)
    list(REMOVE_ITEM SUPPORTED_CUDA_ARCHITECTURES "70")
endif()

if(NVIDIA_CMAKE_BUILD_FOR_DETECTED_ARCHS)
    # Auto-detect available GPU compute architectures
    execute_process(COMMAND node -p
                    "require('@nvidia/rapids-core').cmake_modules_path"
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    OUTPUT_VARIABLE NVIDIA_CMAKE_MODULES_PATH
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    include(${NVIDIA_CMAKE_MODULES_PATH}/EvalGpuArchs.cmake)
    evaluate_gpu_archs(CMAKE_CUDA_ARCHITECTURES)
    list(TRANSFORM CMAKE_CUDA_ARCHITECTURES APPEND "-real")
elseif(NVIDIA_CMAKE_BUILD_FOR_ALL_CUDA_ARCHS)
    set(CMAKE_CUDA_ARCHITECTURES ${SUPPORTED_CUDA_ARCHITECTURES})
    # CMake architecture list entry of "80" means to build compute and sm.
    # What we want is for the newest arch only to build that way
    # while the rest built only for sm.
    list(SORT CMAKE_CUDA_ARCHITECTURES ORDER ASCENDING)
    list(POP_BACK CMAKE_CUDA_ARCHITECTURES latest_arch)
    list(TRANSFORM CMAKE_CUDA_ARCHITECTURES APPEND "-real")
    list(APPEND CMAKE_CUDA_ARCHITECTURES ${latest_arch})
endif()

message(STATUS "BUILD_FOR_DETECTED_ARCHS: ${NVIDIA_CMAKE_BUILD_FOR_DETECTED_ARCHS}")
message(STATUS "BUILD_FOR_ALL_CUDA_ARCHS: ${NVIDIA_CMAKE_BUILD_FOR_ALL_CUDA_ARCHS}")
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
