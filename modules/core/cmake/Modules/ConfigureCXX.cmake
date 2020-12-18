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

###################################################################################################
# - CMake properties ------------------------------------------------------------------------------

if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

if(NVIDIA_USE_CCACHE)
    find_program(CCACHE_PROGRAM_PATH ccache)
    if(CCACHE_PROGRAM_PATH)
        message(STATUS "Using ccache: ${CCACHE_PROGRAM_PATH}")
        set(CCACHE_COMMAND CACHE STRING "${CCACHE_PROGRAM_PATH}")
        if(DEFINED ENV{CCACHE_DIR})
            message(STATUS "Using ccache directory: $ENV{CCACHE_DIR}")
            set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CCACHE_PROGRAM_PATH}")
            set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM_PATH}")
        else()
            execute_process(COMMAND node -p
                            "require('@nvidia/rapids-core').modules_path"
                            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                            OUTPUT_VARIABLE NVIDIA_MODULES_BASE_DIR
                            OUTPUT_STRIP_TRAILING_WHITESPACE)
            execute_process(COMMAND node -p
                            "require('@nvidia/rapids-core').cmake_modules_path"
                            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                            OUTPUT_VARIABLE NVIDIA_CMAKE_MODULES_PATH
                            OUTPUT_STRIP_TRAILING_WHITESPACE)
            execute_process(COMMAND node -p
                            "require('@nvidia/rapids-core').ccache_path"
                            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                            OUTPUT_VARIABLE NVIDIA_CMAKE_CCACHE_DIR
                            OUTPUT_STRIP_TRAILING_WHITESPACE)
            message(STATUS "Using ccache directory: ${NVIDIA_CMAKE_CCACHE_DIR}")
            # Write or update the ccache configuration file
            configure_file("${NVIDIA_CMAKE_MODULES_PATH}/ccache.conf.in" "${NVIDIA_CMAKE_CCACHE_DIR}/ccache.conf")
            set(ENV{CCACHE_CONFIGPATH} "${NVIDIA_CMAKE_CCACHE_DIR}/ccache.conf")
            set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK
                "CCACHE_CONFIGPATH=${NVIDIA_CMAKE_CCACHE_DIR}/ccache.conf ${CCACHE_PROGRAM_PATH}")
            set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE
                "CCACHE_CONFIGPATH=${NVIDIA_CMAKE_CCACHE_DIR}/ccache.conf ${CCACHE_PROGRAM_PATH}")
        endif(DEFINED ENV{CCACHE_DIR})
    endif(CCACHE_PROGRAM_PATH)
endif(NVIDIA_USE_CCACHE)

execute_process(COMMAND node -p
                "require('@nvidia/rapids-core').cpm_source_cache_path"
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE NVIDIA_CPM_SOURCE_CACHE
                OUTPUT_STRIP_TRAILING_WHITESPACE)

set(ENV{CPM_SOURCE_CACHE} ${NVIDIA_CPM_SOURCE_CACHE})
message(STATUS "Using CPM source cache: $ENV{CPM_SOURCE_CACHE}")

execute_process(COMMAND node -p
                "require('@nvidia/rapids-core').cpp_include_path"
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE RAPIDS_CORE_INCLUDE_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE)

message(STATUS "RAPIDS core include: ${RAPIDS_CORE_INCLUDE_DIR}")

###################################################################################################
# - compiler options ------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX)
    option(CMAKE_CXX11_ABI "Enable the GLIBCXX11 ABI" ON)
    string(APPEND CMAKE_CXX_FLAGS " -Werror -Wno-error=deprecated-declarations")
    if(CMAKE_CXX11_ABI)
        message(STATUS "Enabling the GLIBCXX11 ABI")
    else()
        message(STATUS "Disabling the GLIBCXX11 ABI")
        string(APPEND CMAKE_C_FLAGS " -D_GLIBCXX_USE_CXX11_ABI=0")
        string(APPEND CMAKE_CXX_FLAGS " -D_GLIBCXX_USE_CXX11_ABI=0")
        string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler=-D_GLIBCXX_USE_CXX11_ABI=0")
    endif(CMAKE_CXX11_ABI)
endif(CMAKE_COMPILER_IS_GNUCXX)

if(WIN32)
    string(APPEND CMAKE_C_FLAGS " -D_WIN32")
    string(APPEND CMAKE_CXX_FLAGS " -D_WIN32")
    string(APPEND CMAKE_CUDA_FLAGS " -D_WIN32")
elseif(LINUX)
    string(APPEND CMAKE_C_FLAGS " -D__linux__")
    string(APPEND CMAKE_CXX_FLAGS " -D__linux__")
    string(APPEND CMAKE_CUDA_FLAGS " -D__linux__")
elseif(APPLE)
    string(APPEND CMAKE_C_FLAGS " -D__APPLE__")
    string(APPEND CMAKE_CXX_FLAGS " -D__APPLE__")
    string(APPEND CMAKE_CUDA_FLAGS " -D__APPLE__")
endif()

if(DISABLE_DEPRECATION_WARNINGS)
    string(APPEND CMAKE_C_FLAGS " -Wno-deprecated-declarations")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-deprecated-declarations")
    string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler=-Wno-deprecated-declarations")
endif(DISABLE_DEPRECATION_WARNINGS)

string(APPEND CMAKE_C_FLAGS " -fdiagnostics-color=always")
string(APPEND CMAKE_CXX_FLAGS " -fdiagnostics-color=always")
string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler=-fdiagnostics-color=always")
