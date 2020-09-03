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

option(CUDA_HOME "Path to the current CUDA Toolkit root (e.g. /usr/local/cuda)" $ENV{CUDA_HOME})
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
