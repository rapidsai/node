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

project(node_cudf VERSION 0.0.1 LANGUAGES C CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(NVIDIA_USE_CCACHE "Enable caching compilation results with ccache" ON)
option(DISABLE_DEPRECATION_WARNINGS "Disable warnings generated from deprecated declarations." ON)

set(RMM_VERSION "0.18")
set(CUDF_VERSION "0.18")

###################################################################################################
# - cmake modules ---------------------------------------------------------------------------------

execute_process(COMMAND node -p
                "require('@nvidia/rapids-core').cmake_modules_path"
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE NVIDIA_CMAKE_MODULES_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(COMMAND node -p
                "require('path').dirname(require.resolve('@nvidia/cuda'))"
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE NVIDIA_CUDA_MODULE_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(COMMAND node -p
                "require('path').dirname(require.resolve('@nvidia/rmm'))"
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE NVIDIA_RMM_MODULE_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${NVIDIA_CMAKE_MODULES_PATH}")

include(ConfigureCXX)
include(ConfigureCUDA)
include(ConfigureNapi)
include(ConfigureRMM)
include(ConfigureCUDF)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src
                    ${NVIDIA_RMM_MODULE_PATH}/src
                    ${NVIDIA_CUDA_MODULE_PATH}/src
                    ${RAPIDS_CORE_INCLUDE_DIR}
                    ${CUDF_INCLUDE_DIRS}
                    ${NAPI_INCLUDE_DIRS})
                    
###################################################################################################
# - source paths ----------------------------------------------------------------------------------

file(GLOB_RECURSE NODE_CUDF_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

if(cudf_ADDED)
    list(
        APPEND NODE_CUDF_SRC_FILES
        "${cudf_SOURCE_DIR}/cpp/src/bitmask/null_mask.cu"
        "${cudf_SOURCE_DIR}/cpp/src/copying/get_element.cu"
        "${cudf_SOURCE_DIR}/cpp/src/column/column.cu"
        "${cudf_SOURCE_DIR}/cpp/src/column/column_view.cpp"
        "${cudf_SOURCE_DIR}/cpp/src/column/column_factories.cpp"
        "${cudf_SOURCE_DIR}/cpp/src/column/column_device_view.cu"
        "${cudf_SOURCE_DIR}/cpp/src/reductions/minmax.cu"
        "${cudf_SOURCE_DIR}/cpp/src/scalar/scalar.cpp"
        "${cudf_SOURCE_DIR}/cpp/src/scalar/scalar_factories.cpp"
        "${cudf_SOURCE_DIR}/cpp/src/table/table.cpp"
        "${cudf_SOURCE_DIR}/cpp/src/table/table_view.cpp"
    )
endif()

add_library(${PROJECT_NAME} SHARED ${NODE_CUDF_SRC_FILES} ${CMAKE_JS_SRC})

set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")

target_link_libraries(${PROJECT_NAME}
                      ${CMAKE_JS_LIB}
                      rmm::rmm
                      CUDA::nvrtc
                      CUDA::cuda_driver
                      CUDA::cudart_static
                      "${NVIDIA_RMM_MODULE_PATH}/build/${CMAKE_BUILD_TYPE}/node_rmm.node"
                      "${NVIDIA_CUDA_MODULE_PATH}/build/${CMAKE_BUILD_TYPE}/node_cuda.node")
