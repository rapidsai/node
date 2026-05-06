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
include_guard(GLOBAL)

if(NOT DEFINED ENV{NODE_RAPIDS_USE_LOCAL_DEPS_BUILD_DIRS})
    execute_process(COMMAND node -p
                    "require('@rapidsai/core').cpm_source_cache_path"
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                    OUTPUT_VARIABLE NODE_RAPIDS_CPM_SOURCE_CACHE
                    OUTPUT_STRIP_TRAILING_WHITESPACE)

    set(CPM_USE_LOCAL_PACKAGES ON)
    set(CPM_USE_LOCAL_PACKAGES ${CPM_USE_LOCAL_PACKAGES} CACHE BOOL "" FORCE)
    set(ENV{CPM_USE_LOCAL_PACKAGES} ON)

    set(CPM_SOURCE_CACHE "${NODE_RAPIDS_CPM_SOURCE_CACHE}")
    set(CPM_SOURCE_CACHE "${CPM_SOURCE_CACHE}" CACHE STRING "" FORCE)
    set(ENV{CPM_SOURCE_CACHE} "${NODE_RAPIDS_CPM_SOURCE_CACHE}")
    message(VERBOSE "get_cpm: Using CPM source cache: $ENV{CPM_SOURCE_CACHE}")

    execute_process(COMMAND node -p
                    "require('@rapidsai/core').cpm_binary_cache_path"
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                    OUTPUT_VARIABLE NODE_RAPIDS_CPM_BINARY_CACHE
                    OUTPUT_STRIP_TRAILING_WHITESPACE)

    set(CPM_BINARY_CACHE "${NODE_RAPIDS_CPM_BINARY_CACHE}/${CMAKE_BUILD_TYPE}")
    set(CPM_BINARY_CACHE "${CPM_BINARY_CACHE}" CACHE STRING "" FORCE)
    set(ENV{CPM_BINARY_CACHE} "${CPM_BINARY_CACHE}")
    message(VERBOSE "get_cpm: Using CPM BINARY cache: $ENV{CPM_BINARY_CACHE}")

    message(VERBOSE "get_cpm: Using CMake FetchContent base dir: ${CPM_BINARY_CACHE}")
    set(FETCHCONTENT_BASE_DIR "${CPM_BINARY_CACHE}" CACHE STRING "" FORCE)
endif()

function(_clean_build_dirs_if_not_fully_built dir libname)
    if(NOT DEFINED ENV{NODE_RAPIDS_USE_LOCAL_DEPS_BUILD_DIRS})
        if(EXISTS "${CPM_BINARY_CACHE}/${dir}-build/${libname}.a")
            message(VERBOSE "get_cpm: not clearing shared build dirs since '${CPM_BINARY_CACHE}/${dir}-build/${libname}.a' exists")
        elseif(EXISTS "${CPM_BINARY_CACHE}/${dir}-build/${libname}.so")
            message(VERBOSE "get_cpm: not clearing shared build dirs since '${CPM_BINARY_CACHE}/${dir}-build/${libname}.so' exists")
        else()
            file(REMOVE_RECURSE "${CPM_BINARY_CACHE}/${dir}-build")
            file(REMOVE_RECURSE "${CPM_BINARY_CACHE}/${dir}-subbuild")
            message(STATUS "get_cpm: clearing shared build dirs since '${CPM_BINARY_CACHE}/${dir}-build/${libname}.(a|so)' does not exist")
        endif()
    endif()
endfunction()

function(_get_update_disconnected_state target version out_var)
    # We only want to set `UPDATE_DISCONNECTED` while
    # the GIT tag hasn't moved from the last time we cloned
    set(cpm_${target}_disconnect_update "UPDATE_DISCONNECTED TRUE")
    set(cpm_${target}_CURRENT_VERSION ${version} CACHE STRING "version of ${target} we checked out" PARENT_SCOPE)
    if(NOT VERSION VERSION_EQUAL cpm_${target}_CURRENT_VERSION)
        set(cpm_${target}_CURRENT_VERSION ${version} CACHE STRING "version of ${target} we checked out" FORCE PARENT_SCOPE)
        set(cpm_${target}_disconnect_update "")
    endif()
    set(${out_var} cpm_${target}_disconnect_update PARENT_SCOPE)
endfunction()

include(${CMAKE_CURRENT_LIST_DIR}/get_version.cmake)
_get_rapidsai_module_version(rapids-cmake rapids-cmake-version rapids-cmake-branch)

set(_rapids_cmake_init_path "${CMAKE_BINARY_DIR}/RAPIDS.cmake")

if(NOT DEFINED ENV{NODE_RAPIDS_USE_LOCAL_DEPS_BUILD_DIRS})
  set(_rapids_cmake_init_path "${CPM_BINARY_CACHE}/RAPIDS.cmake")
endif()

if(NOT EXISTS "${_rapids_cmake_init_path}")
  message(STATUS "Downloading https://raw.githubusercontent.com/rapidsai/rapids-cmake/${rapids-cmake-branch}/RAPIDS.cmake to ${_rapids_cmake_init_path}")
  file(
    DOWNLOAD
      "https://raw.githubusercontent.com/rapidsai/rapids-cmake/${rapids-cmake-branch}/RAPIDS.cmake"
      "${_rapids_cmake_init_path}"
    SHOW_PROGRESS
    STATUS _rapids_cmake_download_versioned_result
  )

  list(POP_FRONT _rapids_cmake_download_versioned_result _rapids_cmake_download_versioned_status)

  if(NOT "${_rapids_cmake_download_versioned_status}" STREQUAL "0")
    list(POP_FRONT _rapids_cmake_download_versioned_result _rapids_cmake_download_versioned_message)
    message(FATAL_ERROR "Failed to download rapids-cmake@${rapids-cmake-branch}/RAPIDS.cmake:\n HTTP STATUS ${_rapids_cmake_download_versioned_status}: ${_rapids_cmake_download_versioned_message}")
  endif()
endif()

message(STATUS "get_cpm: rapids-cmake-version: ${rapids-cmake-version}")

include("${_rapids_cmake_init_path}")
include(rapids-cmake)
include(rapids-cpm)
include(rapids-find)
include(rapids-export)
include("${rapids-cmake-dir}/export/find_package_root.cmake")

rapids_cpm_init()
