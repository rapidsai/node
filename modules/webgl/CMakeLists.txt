#=============================================================================
# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
cmake_minimum_required(VERSION 3.24.1 FATAL_ERROR)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
unset(CMAKE_LIBRARY_OUTPUT_DIRECTORY)
unset(CMAKE_LIBRARY_OUTPUT_DIRECTORY CACHE)

option(NODE_RAPIDS_WEBGL_STATIC_LINK "Statically link GLEW libraries" ON)
option(NODE_RAPIDS_USE_SCCACHE "Enable caching compilation results with sccache" ON)

###################################################################################################
# - cmake modules ---------------------------------------------------------------------------------

execute_process(COMMAND node -p
                "require('@rapidsai/core').cmake_modules_path"
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                OUTPUT_VARIABLE NODE_RAPIDS_CMAKE_MODULES_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE)

include("${NODE_RAPIDS_CMAKE_MODULES_PATH}/cmake_policies.cmake")

project(rapidsai_webgl VERSION $ENV{npm_package_version} LANGUAGES C CXX)

execute_process(COMMAND node -p
                "require('path').dirname(require.resolve('@rapidsai/core'))"
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                OUTPUT_VARIABLE NODE_RAPIDS_CORE_MODULE_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE)

include("${NODE_RAPIDS_CMAKE_MODULES_PATH}/ConfigureCXX.cmake")
include("${NODE_RAPIDS_CMAKE_MODULES_PATH}/ConfigureNapi.cmake")
include("${NODE_RAPIDS_CMAKE_MODULES_PATH}/ConfigureOpenGL.cmake")
include("${NODE_RAPIDS_CMAKE_MODULES_PATH}/ConfigureOpenGLEW.cmake")
include("${NODE_RAPIDS_CMAKE_MODULES_PATH}/install_utils.cmake")

find_and_configure_glew(
    VERSION 2.1.0
    USE_STATIC ${NODE_RAPIDS_WEBGL_STATIC_LINK}
    EXPORT_SET rapidsai_webgl-exports)

###################################################################################################
# - rapidsai_webgl target -------------------------------------------------------------------------

file(GLOB_RECURSE NODE_WEBGL_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(${PROJECT_NAME} SHARED ${NODE_WEBGL_SRC_FILES} ${CMAKE_JS_SRC})

set_target_properties(${PROJECT_NAME}
    PROPERTIES PREFIX                              ""
               SUFFIX                              ".node"
               BUILD_RPATH                         "\$ORIGIN"
               INSTALL_RPATH                       "\$ORIGIN"
               CXX_STANDARD                        17
               CXX_STANDARD_REQUIRED               ON
               CUDA_STANDARD                       17
               CUDA_STANDARD_REQUIRED              ON
               POSITION_INDEPENDENT_CODE           ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_compile_options(${PROJECT_NAME}
            PRIVATE "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:C>:${NODE_RAPIDS_CMAKE_C_FLAGS}>>"
                    "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:${NODE_RAPIDS_CMAKE_CXX_FLAGS}>>"
)

target_include_directories(${PROJECT_NAME}
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
           "$<BUILD_INTERFACE:${RAPIDS_CORE_INCLUDE_DIR}>"
           "$<BUILD_INTERFACE:${NAPI_INCLUDE_DIRS}>"
)

target_link_libraries(${PROJECT_NAME}
               PUBLIC ${CMAKE_JS_LIB}
                      ${GLEW_LIBRARY}
                      OpenGL::EGL
                      OpenGL::OpenGL
                      "${NODE_RAPIDS_CORE_MODULE_PATH}/build/${CMAKE_BUILD_TYPE}/rapidsai_core.node")

generate_install_rules(NAME ${PROJECT_NAME})

# Create a symlink to compile_commands.json for the llvm-vs-code-extensions.vscode-clangd plugin
execute_process(COMMAND
    ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json)
