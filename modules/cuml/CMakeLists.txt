#=============================================================================
# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

project(rapidsai_cuml VERSION 0.0.1 LANGUAGES C CXX)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    return()
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
unset(CMAKE_LIBRARY_OUTPUT_DIRECTORY)
unset(CMAKE_LIBRARY_OUTPUT_DIRECTORY CACHE)

option(NODE_RAPIDS_USE_SCCACHE "Enable caching compilation results with sccache" ON)

###################################################################################################
# - cmake modules ---------------------------------------------------------------------------------

execute_process(COMMAND node -p
                "require('@rapidsai/core').cmake_modules_path"
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                OUTPUT_VARIABLE NODE_RAPIDS_CMAKE_MODULES_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(COMMAND node -p
                "require('path').dirname(require.resolve('@rapidsai/core'))"
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                OUTPUT_VARIABLE NODE_RAPIDS_CORE_MODULE_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(COMMAND node -p
                "require('path').dirname(require.resolve('@rapidsai/cuda'))"
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                OUTPUT_VARIABLE NODE_RAPIDS_CUDA_MODULE_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(COMMAND node -p
                "require('path').dirname(require.resolve('@rapidsai/rmm'))"
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                OUTPUT_VARIABLE NODE_RAPIDS_RMM_MODULE_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(COMMAND node -p
                "require('path').dirname(require.resolve('@rapidsai/cudf'))"
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                OUTPUT_VARIABLE NODE_RAPIDS_CUDF_MODULE_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE)

include("${NODE_RAPIDS_CMAKE_MODULES_PATH}/ConfigureCXX.cmake")
include("${NODE_RAPIDS_CMAKE_MODULES_PATH}/ConfigureCUDA.cmake")
include("${NODE_RAPIDS_CMAKE_MODULES_PATH}/ConfigureNapi.cmake")

if(NOT DEFINED ENV{NODE_RAPIDS_USE_LOCAL_DEPS_BUILD_DIRS})
    include("${NODE_RAPIDS_CMAKE_MODULES_PATH}/ConfigureCUDF.cmake")
endif()

include("${NODE_RAPIDS_CMAKE_MODULES_PATH}/ConfigureCUML.cmake")

###################################################################################################
# - rapidsai_cuml target ---------------------------------------------------------------------------

file(GLOB_RECURSE NODE_CUML_CPP_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE NODE_CUML_CUDA_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")

list(APPEND NODE_CUML_SRC_FILES ${NODE_CUML_CPP_FILES})
list(APPEND NODE_CUML_SRC_FILES ${NODE_CUML_CUDA_FILES})

add_library(${PROJECT_NAME} SHARED ${NODE_CUML_SRC_FILES} ${CMAKE_JS_SRC})

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

if(TARGET cudf::cudf)
    set(LIBCUDF_LIBRARY cudf::cudf)
    set(LIBCUDF_INCLUDE_DIRS "")
else()
    set(LIBCUDF_LIBRARY "${NODE_RAPIDS_CUDF_MODULE_PATH}/build/${CMAKE_BUILD_TYPE}/_deps/cudf-build/libcudf.so")
    list(APPEND LIBCUDF_INCLUDE_DIRS "${NODE_RAPIDS_CUDF_MODULE_PATH}/build/${CMAKE_BUILD_TYPE}/_deps/cudf-build/include")
    list(APPEND LIBCUDF_INCLUDE_DIRS "${NODE_RAPIDS_CUDF_MODULE_PATH}/build/${CMAKE_BUILD_TYPE}/_deps/cudf-src/cpp/include")
endif()

target_include_directories(${PROJECT_NAME}
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
           "$<BUILD_INTERFACE:${NODE_RAPIDS_CUDF_MODULE_PATH}/src>"
           "$<BUILD_INTERFACE:${NODE_RAPIDS_RMM_MODULE_PATH}/src>"
           "$<BUILD_INTERFACE:${NODE_RAPIDS_CUDA_MODULE_PATH}/src>"
           "$<BUILD_INTERFACE:${RAPIDS_CORE_INCLUDE_DIR}>"
           "$<BUILD_INTERFACE:${LIBCUDF_INCLUDE_DIRS}>"
           "$<BUILD_INTERFACE:${NAPI_INCLUDE_DIRS}>"
)

target_link_libraries(${PROJECT_NAME}
               PUBLIC ${CMAKE_JS_LIB}
                      cuml::cuml++
                      "${LIBCUDF_LIBRARY}"
                      "${NODE_RAPIDS_CUDF_MODULE_PATH}/build/${CMAKE_BUILD_TYPE}/rapidsai_cudf.node"
                      "${NODE_RAPIDS_RMM_MODULE_PATH}/build/${CMAKE_BUILD_TYPE}/rapidsai_rmm.node"
                      "${NODE_RAPIDS_CUDA_MODULE_PATH}/build/${CMAKE_BUILD_TYPE}/rapidsai_cuda.node"
                      "${NODE_RAPIDS_CORE_MODULE_PATH}/build/${CMAKE_BUILD_TYPE}/rapidsai_core.node")

# Create a symlink to compile_commands.json for the llvm-vs-code-extensions.vscode-clangd plugin
execute_process(COMMAND
    ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json)
