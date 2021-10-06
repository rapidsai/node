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

cmake_minimum_required(VERSION 3.18...3.20 FATAL_ERROR)

project(rapidsai_sql VERSION 0.0.1 LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
unset(CMAKE_LIBRARY_OUTPUT_DIRECTORY)
unset(CMAKE_LIBRARY_OUTPUT_DIRECTORY CACHE)

option(NODE_RAPIDS_USE_SCCACHE "Enable caching compilation results with sccache" ON)
option(DISABLE_DEPRECATION_WARNINGS "Disable warnings generated from deprecated declarations." ON)

set(RMM_VERSION "21.12.00")
set(CUDF_VERSION "21.12.00")
set(BLAZINGSQL_VERSION "21.12.00")

###################################################################################################
# - cmake modules ---------------------------------------------------------------------------------

execute_process(COMMAND node -p
                "require('@rapidsai/core').cmake_modules_path"
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE NODE_RAPIDS_CMAKE_MODULES_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(COMMAND node -p
                "require('path').dirname(require.resolve('@nvidia/cuda'))"
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE NODE_RAPIDS_CUDA_MODULE_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(COMMAND node -p
                "require('path').dirname(require.resolve('@rapidsai/rmm'))"
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE NODE_RAPIDS_RMM_MODULE_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(COMMAND node -p
                "require('path').dirname(require.resolve('@rapidsai/cudf'))"
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE NODE_RAPIDS_CUDF_MODULE_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE)

list(APPEND CMAKE_MODULE_PATH "${NODE_RAPIDS_CMAKE_MODULES_PATH}")

include(ConfigureCXX)
include(ConfigureCUDA)
include(ConfigureNapi)
include(ConfigureBlazingSQL)

###################################################################################################
# - blazingsql_wrapper target ------------------------------------------------------------------------------

set(BLAZINGSQL_WRAPPER_CMAKE_C_FLAGS ${NODE_RAPIDS_CMAKE_C_FLAGS})
set(BLAZINGSQL_WRAPPER_CMAKE_CXX_FLAGS ${NODE_RAPIDS_CMAKE_CXX_FLAGS})
set(BLAZINGSQL_WRAPPER_CMAKE_CUDA_FLAGS ${NODE_RAPIDS_CMAKE_CUDA_FLAGS})

# Silence warnings
list(APPEND BLAZINGSQL_WRAPPER_CMAKE_C_FLAGS -w)
list(APPEND BLAZINGSQL_WRAPPER_CMAKE_CXX_FLAGS -w)
list(APPEND BLAZINGSQL_WRAPPER_CMAKE_CUDA_FLAGS -w)

# Remove -Werror from flags because blazingsql headers are messy
list(REMOVE_ITEM BLAZINGSQL_WRAPPER_CMAKE_C_FLAGS -Werror)
list(REMOVE_ITEM BLAZINGSQL_WRAPPER_CMAKE_CXX_FLAGS -Werror)
list(REMOVE_ITEM BLAZINGSQL_WRAPPER_CMAKE_CUDA_FLAGS -Werror)

file(GLOB_RECURSE BLAZINGSQL_WRAPPER_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/blazingsql/*.cpp")

add_library(blazingsql_wrapper SHARED ${BLAZINGSQL_WRAPPER_SRC_FILES})

set_target_properties(blazingsql_wrapper
    PROPERTIES PREFIX                              ""
               SUFFIX                              ".so"
               BUILD_RPATH                         "\$ORIGIN"
               INSTALL_RPATH                       "\$ORIGIN"
               CXX_STANDARD                        17
               CXX_STANDARD_REQUIRED               ON
               CUDA_STANDARD                       17
               CUDA_STANDARD_REQUIRED              ON
               NO_SYSTEM_FROM_IMPORTED             ON
               POSITION_INDEPENDENT_CODE           ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
               LIBRARY_OUTPUT_DIRECTORY            lib
)

target_compile_options(blazingsql_wrapper
            PRIVATE "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:C>:${BLAZINGSQL_WRAPPER_CMAKE_C_FLAGS}>>"
                    "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:${BLAZINGSQL_WRAPPER_CMAKE_CXX_FLAGS}>>"
                    "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CUDA>:${BLAZINGSQL_WRAPPER_CMAKE_CUDA_FLAGS}>>"
)

target_include_directories(blazingsql_wrapper
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/blazingsql>"
           "$<BUILD_INTERFACE:${NODE_RAPIDS_CUDF_MODULE_PATH}/src>"
           "$<BUILD_INTERFACE:${NODE_RAPIDS_RMM_MODULE_PATH}/src>"
           "$<BUILD_INTERFACE:${NODE_RAPIDS_CUDA_MODULE_PATH}/src>"
           "$<BUILD_INTERFACE:${RAPIDS_CORE_INCLUDE_DIR}>"
           "$<BUILD_INTERFACE:${NAPI_INCLUDE_DIRS}>"
)

target_link_libraries(blazingsql_wrapper
    PUBLIC  rmm::rmm
            cudf::cudf
    PRIVATE blazingdb::blazingsql-engine
    PUBLIC  "${NODE_RAPIDS_CUDF_MODULE_PATH}/build/${CMAKE_BUILD_TYPE}/node_cudf.node"
            "${NODE_RAPIDS_RMM_MODULE_PATH}/build/${CMAKE_BUILD_TYPE}/node_rmm.node"
            "${NODE_RAPIDS_CUDA_MODULE_PATH}/build/${CMAKE_BUILD_TYPE}/node_cuda.node"
)

###################################################################################################
# - rapidsai_sql target ------------------------------------------------------------------------------

file(GLOB_RECURSE NODE_BLAZINGSQL_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(${PROJECT_NAME} SHARED ${NODE_BLAZINGSQL_SRC_FILES} ${CMAKE_JS_SRC})

set_target_properties(${PROJECT_NAME}
    PROPERTIES PREFIX                              ""
               SUFFIX                              ".node"
               BUILD_RPATH                         "\$ORIGIN"
               INSTALL_RPATH                       "\$ORIGIN"
               CXX_STANDARD                        17
               CXX_STANDARD_REQUIRED               ON
               CUDA_STANDARD                       17
               CUDA_STANDARD_REQUIRED              ON
               NO_SYSTEM_FROM_IMPORTED             ON
               POSITION_INDEPENDENT_CODE           ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_compile_options(${PROJECT_NAME}
            PRIVATE "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:C>:${NODE_RAPIDS_CMAKE_C_FLAGS}>>"
                    "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:${NODE_RAPIDS_CMAKE_CXX_FLAGS}>>"
                    "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CUDA>:${NODE_RAPIDS_CMAKE_CUDA_FLAGS}>>"
)

target_include_directories(${PROJECT_NAME}
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
           "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/blazingsql>"
           "$<BUILD_INTERFACE:${NODE_RAPIDS_CUDF_MODULE_PATH}/src>"
           "$<BUILD_INTERFACE:${NODE_RAPIDS_RMM_MODULE_PATH}/src>"
           "$<BUILD_INTERFACE:${NODE_RAPIDS_CUDA_MODULE_PATH}/src>"
           "$<BUILD_INTERFACE:${RAPIDS_CORE_INCLUDE_DIR}>"
           "$<BUILD_INTERFACE:${NAPI_INCLUDE_DIRS}>"
)

target_compile_definitions(${PROJECT_NAME} PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM)

target_link_libraries(${PROJECT_NAME}
                      ${CMAKE_JS_LIB}
                      rmm::rmm
                      cudf::cudf
                      blazingsql_wrapper
                      ucp
                      "${NODE_RAPIDS_CUDF_MODULE_PATH}/build/${CMAKE_BUILD_TYPE}/node_cudf.node"
                      "${NODE_RAPIDS_RMM_MODULE_PATH}/build/${CMAKE_BUILD_TYPE}/node_rmm.node"
                      "${NODE_RAPIDS_CUDA_MODULE_PATH}/build/${CMAKE_BUILD_TYPE}/node_cuda.node")

# Create a symlink to compile_commands.json for the llvm-vs-code-extensions.vscode-clangd plugin
execute_process(COMMAND
    ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json)
