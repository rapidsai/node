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

#Very important the first step is to enable the CUDA language.
enable_language(CUDA)

if(NOT CMAKE_CUDA_COMPILER)
  message(FATAL_ERROR "CMake cannot locate a CUDA compiler")
endif()

set(CUDA_HOME "Path to the current CUDA Toolkit root (e.g. /usr/local/cuda)" $ENV{CUDA_HOME})
option(CUDA_SEPARABLE_COMPILATION "Compile CUDA objects with separable compilation enabled.  Requires CUDA 5.0+" OFF)

if(CMAKE_CUDA_COMPILER_VERSION)
    # Compute the version. from  CMAKE_CUDA_COMPILER_VERSION
    string(REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\1" CUDA_VERSION_MAJOR ${CMAKE_CUDA_COMPILER_VERSION})
    string(REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\2" CUDA_VERSION_MINOR ${CMAKE_CUDA_COMPILER_VERSION})
    set(CUDA_VERSION "${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}" CACHE STRING "Version of CUDA as computed from nvcc.")
    mark_as_advanced(CUDA_VERSION)
endif()

# Always set this convenience variable
set(CUDA_VERSION_STRING "${CUDA_VERSION}")
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CUDA_VERSION VERSION_LESS "9.2")
  message(FATAL_ERROR "Unsupported CUDA toolkit version ${CUDA_VERSION_STRING}")
endif()

# Set the CUDA_NVCC_EXECUTABLE
set(CUDA_NVCC_EXECUTABLE "${CMAKE_CUDA_COMPILER}")

# Find the CUDA_LIBRARIES like FindCUDA does
find_path(CUDA_LIBRARIES NAMES libcudart.so cudart.lib # CUDA toolkit runtime library
    PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
    PATH_SUFFIXES lib NO_DEFAULT_PATH)

# Find the CUDA_INCLUDE_DIRS like FindCUDA does
find_path(CUDA_INCLUDE_DIRS NAMES cuda.h cuda_runtime.h device_functions.h # CUDA toolkit headers
    PATHS ${CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES}
    PATH_SUFFIXES include NO_DEFAULT_PATH
    HINTS "/usr/local/cuda-${CUDA_VERSION}/include"
          "/usr/local/cuda-${CUDA_VERSION}/targets/x86_64-linux/include")

string(APPEND CMAKE_CUDA_FLAGS " --expt-extended-lambda --expt-relaxed-constexpr")
string(APPEND CMAKE_CUDA_FLAGS " -Werror=cross-execution-space-call")

# Auto-detect available GPU compute architectures
set(CUDA_ARCHITECTURES "$ENV{CUDA_ARCHITECTURES}" CACHE STRING
  "List of GPU architectures (semicolon-separated) to be compiled for. Pass 'ALL' if you want to compile for all supported GPU architectures. Empty string means to auto-detect the GPUs on the current system")

if("${CUDA_ARCHITECTURES}" STREQUAL "")
  execute_process(COMMAND node -p
                  "require('@nvidia/rapids-core').cmake_modules_path"
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  OUTPUT_VARIABLE NVIDIA_CMAKE_MODULES_PATH
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  include(${NVIDIA_CMAKE_MODULES_PATH}/EvalGpuArchs.cmake)
  evaluate_gpu_archs(CUDA_ARCHITECTURES)
endif()

if("${CUDA_ARCHITECTURES}" STREQUAL "ALL")
    # This is being built for an x86 or x86_64 architecture
    set(CUDA_ARCHITECTURES "60")
    if((CUDA_VERSION_MAJOR EQUAL 9) OR (CUDA_VERSION_MAJOR GREATER 9))
        list(APPEND CUDA_ARCHITECTURES "70")
    endif()
    if((CUDA_VERSION_MAJOR EQUAL 10) OR (CUDA_VERSION_MAJOR GREATER 10))
        list(APPEND CUDA_ARCHITECTURES "75")
    endif()
    if((CUDA_VERSION_MAJOR EQUAL 11) OR (CUDA_VERSION_MAJOR GREATER 11))
        list(APPEND CUDA_ARCHITECTURES "80")
    endif()
endif()

message("CUDA_ARCHITECTURES: ${CUDA_ARCHITECTURES}")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})
