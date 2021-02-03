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

project(node_rapids_core VERSION 0.0.1 LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

option(NVIDIA_USE_CCACHE "Enable caching compilation results with ccache" ON)
option(DISABLE_DEPRECATION_WARNINGS "Disable warnings generated from deprecated declarations." ON)

# Example
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
                OUTPUT_VARIABLE NVIDIA_CMAKE_MODULES_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${NVIDIA_CMAKE_MODULES_PATH}")

include(ConfigureCXX)
include(ConfigureCUDA)
include(ConfigureNapi)
include(ConfigureRMM)
include(ConfigureCUDF)
include(ConfigureCuGraph)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${CUGRAPH_INCLUDE_DIRS}
                    ${NAPI_INCLUDE_DIRS})

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

file(GLOB_RECURSE NODE_RAPIDS_CORE_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/addon.cpp")

add_library(${PROJECT_NAME} SHARED ${NODE_RAPIDS_CORE_SRC_FILES} ${CMAKE_JS_SRC})

set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")

target_link_libraries(${PROJECT_NAME}
                      ${CMAKE_JS_LIB}
                      rmm::rmm
                      cudf::cudf)
