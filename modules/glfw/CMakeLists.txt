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
cmake_minimum_required(VERSION 3.18...3.20 FATAL_ERROR)

project(rapidsai_glfw VERSION 0.0.1 LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
unset(CMAKE_LIBRARY_OUTPUT_DIRECTORY)
unset(CMAKE_LIBRARY_OUTPUT_DIRECTORY CACHE)

option(NODE_RAPIDS_USE_SCCACHE "Enable caching compilation results with sccache" ON)

# Set OpenGL_GL_PREFERENCE to "GLVND"
cmake_policy(SET CMP0072 NEW)

###################################################################################################
# - cmake modules ---------------------------------------------------------------------------------

execute_process(COMMAND node -p
                "require('@rapidsai/core').cmake_modules_path"
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE NODE_RAPIDS_CMAKE_MODULES_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE)

list(APPEND CMAKE_MODULE_PATH "${NODE_RAPIDS_CMAKE_MODULES_PATH}")

include(ConfigureCXX)
include(ConfigureNapi)
include(ConfigureOpenGL)
include(ConfigureOpenGLFW)

function(_find_glfw VARIANT USE_WAYLAND USE_EGLHEADLESS)
    find_and_configure_glfw(
        glfw_${VARIANT}
        3.3.2                                   # version
        https://github.com/trxcllnt/glfw.git    # git repo
        fea/headless-egl-with-fallback          # git tag
        OFF                                     # build + dynamically link libglfw.so
        ${USE_WAYLAND}
        ${USE_EGLHEADLESS}
    )
endfunction()

_find_glfw(x11 OFF OFF)
# _find_glfw(wayland ON OFF)
_find_glfw(eglheadless OFF ON)

###################################################################################################
# - rapidsai_glfw target ------------------------------------------------------------------------------

file(GLOB_RECURSE NODE_GLFW_SRC_FILES "src/*.cpp")

function(make_rapidsai_glfw_target VARIANT DEFINES)
    add_library(${PROJECT_NAME}_${VARIANT} SHARED ${NODE_GLFW_SRC_FILES} ${CMAKE_JS_SRC})

    set_target_properties(${PROJECT_NAME}_${VARIANT}
        PROPERTIES PREFIX                              ""
                SUFFIX                              ".node"
                BUILD_RPATH                         "\$ORIGIN"
                INSTALL_RPATH                       "\$ORIGIN"
                CXX_STANDARD                        17
                CXX_STANDARD_REQUIRED               ON
                CUDA_STANDARD                       17
                CUDA_STANDARD_REQUIRED              ON
                # NO_SYSTEM_FROM_IMPORTED             ON
                POSITION_INDEPENDENT_CODE           ON
                INTERFACE_POSITION_INDEPENDENT_CODE ON
    )

    set(${PROJECT_NAME}_${VARIANT}_C_FLAGS "")
    set(${PROJECT_NAME}_${VARIANT}_CXX_FLAGS "")
    list(APPEND ${PROJECT_NAME}_${VARIANT}_C_FLAGS ${DEFINES})
    list(APPEND ${PROJECT_NAME}_${VARIANT}_CXX_FLAGS ${DEFINES})
    list(APPEND ${PROJECT_NAME}_${VARIANT}_C_FLAGS ${NODE_RAPIDS_CMAKE_C_FLAGS})
    list(APPEND ${PROJECT_NAME}_${VARIANT}_CXX_FLAGS ${NODE_RAPIDS_CMAKE_CXX_FLAGS})

    target_compile_options(${PROJECT_NAME}_${VARIANT}
        PRIVATE "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:C>:${${PROJECT_NAME}_${VARIANT}_C_FLAGS}>>"
                "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:${${PROJECT_NAME}_${VARIANT}_CXX_FLAGS}>>"
    )

    target_include_directories(${PROJECT_NAME}_${VARIANT}
        PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
               "$<BUILD_INTERFACE:${RAPIDS_CORE_INCLUDE_DIR}>"
               "$<BUILD_INTERFACE:${NAPI_INCLUDE_DIRS}>"
    )

    target_link_libraries(${PROJECT_NAME}_${VARIANT}
                        ${CMAKE_JS_LIB}
                        glfw::${VARIANT}
                        OpenGL::EGL
                        OpenGL::OpenGL)
endfunction()

make_rapidsai_glfw_target(x11 "-DGLFW_EXPOSE_NATIVE_X11")
# make_rapidsai_glfw_target(wayland "-DGLFW_EXPOSE_NATIVE_WAYLAND")
make_rapidsai_glfw_target(eglheadless "-DGLFW_EXPOSE_NATIVE_EGL_EXT -D_GLFW_EGL_LIBRARY=libEGL_nvidia.so.0")

# Create a symlink to compile_commands.json for the llvm-vs-code-extensions.vscode-clangd plugin
execute_process(COMMAND
    ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json)
