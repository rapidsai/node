#=============================================================================
# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

project(node_rmm VERSION 0.0.1 LANGUAGES C CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

option(NODE_RAPIDS_USE_CCACHE "Enable caching compilation results with ccache" ON)
option(DISABLE_DEPRECATION_WARNINGS "Disable warnings generated from deprecated declarations." ON)

set(RMM_VERSION "0.19")

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

list(APPEND CMAKE_MODULE_PATH "${NODE_RAPIDS_CMAKE_MODULES_PATH}")

include(ConfigureCXX)
include(ConfigureCUDA)
include(ConfigureNapi)
include(ConfigureRMM)

###################################################################################################
# - node_rmm target -------------------------------------------------------------------------------

file(GLOB_RECURSE NODE_CUDA_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(${PROJECT_NAME} SHARED ${NODE_CUDA_SRC_FILES} ${CMAKE_JS_SRC})

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
           "$<BUILD_INTERFACE:${NODE_RAPIDS_CUDA_MODULE_PATH}/src>"
           "$<BUILD_INTERFACE:${RAPIDS_CORE_INCLUDE_DIR}>"
           "$<BUILD_INTERFACE:${NAPI_INCLUDE_DIRS}>"
)

target_compile_definitions(${PROJECT_NAME}
    PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:CUDA_API_PER_THREAD_DEFAULT_STREAM>"
           "$<$<COMPILE_LANGUAGE:CUDA>:CUDA_API_PER_THREAD_DEFAULT_STREAM>"
)

target_link_libraries(${PROJECT_NAME}
                      ${CMAKE_JS_LIB}
                      rmm::rmm
                      "${NODE_RAPIDS_CUDA_MODULE_PATH}/build/${CMAKE_BUILD_TYPE}/node_cuda.node")
