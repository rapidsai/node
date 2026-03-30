#=============================================================================
# Copyright (c) 2021-2026, NVIDIA CORPORATION.
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

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

cmake_minimum_required(VERSION 4.2.3 FATAL_ERROR)

execute_process(
  COMMAND node -p "require('@rapidsai/core').cmake_modules_path"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE rapidsai-modules
  COMMAND_ERROR_IS_FATAL ANY
)

include("${rapidsai-modules}/cmake_policies.cmake")

project(rapidsai_io VERSION $ENV{npm_package_version} LANGUAGES C CXX)

###################################################################################################
# - cmake modules ---------------------------------------------------------------------------------

include("${rapidsai-modules}/ConfigureCXX.cmake")
include("${rapidsai-modules}/ConfigureCUDA.cmake")
include("${rapidsai-modules}/ConfigureNapi.cmake")
include("${rapidsai-modules}/install_utils.cmake")

find_packages_and_generate_exports(
  NAME ${PROJECT_NAME}
  NPM_DEPS @rapidsai/cudf
  CPM_DEPS rapidsai::rapidsai_cudf
)

###################################################################################################
# - rapidsai_io target ----------------------------------------------------------------------------

file(GLOB_RECURSE _cpp_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE _cuda_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")

add_library(${PROJECT_NAME} SHARED ${_cpp_sources} ${_cuda_sources} ${CMAKE_JS_SRC})

add_library(rapidsai::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

set_target_properties(${PROJECT_NAME}
    PROPERTIES PREFIX                              ""
               SUFFIX                              ".node"
               BUILD_RPATH                         "\$ORIGIN"
               INSTALL_RPATH                       "\$ORIGIN"
               CXX_STANDARD                        20
               CXX_STANDARD_REQUIRED               ON
               CUDA_STANDARD                       20
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
    PRIVATE "$<BUILD_INTERFACE:${NAPI_INCLUDE_DIRS}>"
  INTERFACE "$<INSTALL_INTERFACE:../../src>"
)

target_link_libraries(${PROJECT_NAME}
     PUBLIC rapidsai::rapidsai_cudf
)

generate_install_rules(
  INSTALL_TOP_LEVEL
  NAME ${PROJECT_NAME}
  GLOBAL_TARGETS ${PROJECT_NAME}
  CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)

# Create a symlink to compile_commands.json for the llvm-vs-code-extensions.vscode-clangd plugin
execute_process(COMMAND
    ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json)
