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
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(node_cugraph VERSION 0.0.1 LANGUAGES C CXX CUDA)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    return()
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

option(NODE_RAPIDS_USE_CCACHE "Enable caching compilation results with ccache" ON)
option(DISABLE_DEPRECATION_WARNINGS "Disable warnings generated from deprecated declarations." ON)

set(RMM_VERSION "0.19")
set(CUDF_VERSION "0.19")
set(CUGRAPH_VERSION "0.19")
set(RAFT_VERSION "0.19")
set(RAFT_BRANCH "f75d7b437bf1da3df749108161b8a0505fb6b7b3")

###################################################################################################
# - cmake modules ---------------------------------------------------------------------------------

execute_process(COMMAND node -p
                "require('@nvidia/rapids-core').cmake_modules_path"
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE NODE_RAPIDS_CMAKE_MODULES_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(COMMAND node -p
                "require('path').dirname(require.resolve('@nvidia/cuda'))"
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE NODE_RAPIDS_CUDA_MODULE_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(COMMAND node -p
                "require('path').dirname(require.resolve('@nvidia/rmm'))"
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE NODE_RAPIDS_RMM_MODULE_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(COMMAND node -p
                "require('path').dirname(require.resolve('@nvidia/cudf'))"
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE NODE_RAPIDS_CUDF_MODULE_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE)

list(APPEND CMAKE_MODULE_PATH "${NODE_RAPIDS_CMAKE_MODULES_PATH}")

include(ConfigureCXX)
include(ConfigureCUDA)
include(ConfigureNapi)
include(ConfigureRMM)
include(ConfigureCUDF)
include(ConfigureCuGraph)

###################################################################################################
# - cuda compiler flags ---------------------------------------------------------------------------

list(APPEND NODE_RAPIDS_CMAKE_CUDA_FLAGS -Xptxas --disable-warnings)
list(APPEND NODE_RAPIDS_CMAKE_CUDA_FLAGS -Xcompiler=-Wall,-Wno-error=sign-compare,-Wno-error=unused-but-set-variable)

###################################################################################################
# - node_cugraph target ---------------------------------------------------------------------------

file(GLOB_RECURSE NODE_CUGRAPH_CPP_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE NODE_CUGRAPH_CUDA_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")

list(APPEND NODE_CUGRAPH_SRC_FILES ${NODE_CUGRAPH_CPP_FILES})
list(APPEND NODE_CUGRAPH_SRC_FILES ${NODE_CUGRAPH_CUDA_FILES})

if(cugraph_SOURCE_DIR)
    list(
        APPEND NODE_CUGRAPH_SRC_FILES
        "${cugraph_SOURCE_DIR}/cpp/src/structure/graph.cu"
        "${cugraph_SOURCE_DIR}/cpp/src/layout/force_atlas2.cu"
    )
endif()

add_library(${PROJECT_NAME} SHARED ${NODE_CUGRAPH_SRC_FILES} ${CMAKE_JS_SRC})

set_target_properties(${PROJECT_NAME}
    PROPERTIES PREFIX                              ""
               SUFFIX                              ".node"
               CXX_STANDARD                        14
               CXX_STANDARD_REQUIRED               ON
               CUDA_STANDARD                       14
               CUDA_STANDARD_REQUIRED              ON
               POSITION_INDEPENDENT_CODE           ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_compile_options(${PROJECT_NAME}
            PRIVATE "$<$<COMPILE_LANGUAGE:C>:${NODE_RAPIDS_CMAKE_C_FLAGS}>"
                    "$<$<COMPILE_LANGUAGE:CXX>:${NODE_RAPIDS_CMAKE_CXX_FLAGS}>"
                    "$<$<COMPILE_LANGUAGE:CUDA>:${NODE_RAPIDS_CMAKE_CUDA_FLAGS}>"
)

target_include_directories(${PROJECT_NAME}
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
           "$<BUILD_INTERFACE:${NODE_RAPIDS_CUDF_MODULE_PATH}/src>"
           "$<BUILD_INTERFACE:${NODE_RAPIDS_RMM_MODULE_PATH}/src>"
           "$<BUILD_INTERFACE:${NODE_RAPIDS_CUDA_MODULE_PATH}/src>"
           "$<BUILD_INTERFACE:${RAPIDS_CORE_INCLUDE_DIR}>"
           "$<BUILD_INTERFACE:${CUGRAPH_INCLUDE_DIRS}>"
           "$<BUILD_INTERFACE:${NAPI_INCLUDE_DIRS}>"
)

target_link_libraries(${PROJECT_NAME}
                      ${CMAKE_JS_LIB}
                      rmm::rmm
                      cudf::cudf
                      CUDA::cuda_driver
                      CUDA::cudart_static
                      "${NODE_RAPIDS_RMM_MODULE_PATH}/build/${CMAKE_BUILD_TYPE}/node_rmm.node"
                      "${NODE_RAPIDS_CUDF_MODULE_PATH}/build/${CMAKE_BUILD_TYPE}/node_cudf.node"
                      "${NODE_RAPIDS_CUDA_MODULE_PATH}/build/${CMAKE_BUILD_TYPE}/node_cuda.node")
