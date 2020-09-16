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

project(node_cugraph VERSION 0.0.1 LANGUAGES C CXX CUDA)

option(NVIDIA_USE_CCACHE "Enable caching compilation results with ccache" ON)

set(RMM_VERSION "0.16")
set(RAFT_VERSION "0.16")
set(CUGRAPH_VERSION "0.16")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

###################################################################################################
# - cmake modules ---------------------------------------------------------------------------------

execute_process(COMMAND node -p
                "require('@nvidia/cmake-modules').cmake_modules_path"
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE NVIDIA_CMAKE_MODULES_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${NVIDIA_CMAKE_MODULES_PATH}")

include(ConfigureCXX)
include(ConfigureCUDA)
include(ConfigureNapi)
include(ConfigureRMM)
include(ConfigureRAFT)
include(ConfigureCuGraph)

###################################################################################################
# - cuda compiler flags ---------------------------------------------------------------------------

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror=cross-execution-space-call -Wno-deprecated-declarations -Xptxas --disable-warnings")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wno-error=sign-compare,-Wno-error=unused-but-set-variable")

set(GPU_ARCHS "${GPU_ARCHS};75")

foreach(arch ${GPU_ARCHS})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${CUDA_INCLUDE_DIRS}"
                    "${RMM_INCLUDE_DIR}"
                    "${RAFT_INCLUDE_DIR}"
                    "${CUGRAPH_INCLUDE_DIR_REAL}"
                    "${CUGRAPH_INCLUDE_DIR}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/include")

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CUDA_LIBRARIES}")

file(GLOB_RECURSE NODE_CUGRAPH_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/*.cpp")

add_library(${PROJECT_NAME} SHARED ${NODE_CUGRAPH_SRC_FILES} ${CMAKE_JS_SRC}
            "${cugraph_SOURCE_DIR}/cpp/src/experimental/graph.cu"
            "${cugraph_SOURCE_DIR}/cpp/src/experimental/graph_view.cu")
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")
target_link_libraries(${PROJECT_NAME}
                      ${CMAKE_JS_LIB} cuda cudart nvrtc rmm)
