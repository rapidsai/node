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

# Very important the first step is to enable the CUDA language.
enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)

string(APPEND CMAKE_CUDA_FLAGS " -Werror=cross-execution-space-call")
string(APPEND CMAKE_CUDA_FLAGS " --expt-extended-lambda --expt-relaxed-constexpr")
string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler=-Wall,-Werror,-Wno-error=deprecated-declarations")

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
    if((CUDAToolkit_VERSION_MAJOR EQUAL 9) OR (CUDAToolkit_VERSION_MAJOR GREATER 9))
        list(APPEND CUDA_ARCHITECTURES "70")
    endif()
    if((CUDAToolkit_VERSION_MAJOR EQUAL 10) OR (CUDAToolkit_VERSION_MAJOR GREATER 10))
        list(APPEND CUDA_ARCHITECTURES "75")
    endif()
    if((CUDAToolkit_VERSION_MAJOR EQUAL 11) OR (CUDAToolkit_VERSION_MAJOR GREATER 11))
        list(APPEND CUDA_ARCHITECTURES "80")
    endif()
endif()

message(STATUS "CUDA_ARCHITECTURES: ${CUDA_ARCHITECTURES}")

set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})
