#=============================================================================
# Copyright (c) 2021, NVIDIA CORPORATION.
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

if(DEFINED CPM_SOURCE_CACHE AND
  (DEFINED ENV{CPM_SOURCE_CACHE}) AND
  (DEFINED CPM_DOWNLOAD_LOCATION))
    message(STATUS "get_cpm: CPM already loaded")
    return()
endif()

execute_process(COMMAND node -p
                "require('@rapidsai/core').cpm_source_cache_path"
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE NODE_RAPIDS_CPM_SOURCE_CACHE
                OUTPUT_STRIP_TRAILING_WHITESPACE)

set(CPM_SOURCE_CACHE "${NODE_RAPIDS_CPM_SOURCE_CACHE}")
set(ENV{CPM_SOURCE_CACHE} "${NODE_RAPIDS_CPM_SOURCE_CACHE}")
message(STATUS "get_cpm: Using CPM source cache: $ENV{CPM_SOURCE_CACHE}")

set(CPM_DOWNLOAD_VERSION 4fad2eac0a3741df3d9c44b791f9163b74aa7b07) # 0.32.0

if(CPM_SOURCE_CACHE)
  # Expand relative path. This is important if the provided path contains a tilde (~)
  get_filename_component(CPM_SOURCE_CACHE ${CPM_SOURCE_CACHE} ABSOLUTE)
  set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
  set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else()
  set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif()

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
  message(VERBOSE "get_cpm: Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
  file(
    DOWNLOAD
    https://raw.githubusercontent.com/cpm-cmake/CPM.cmake/${CPM_DOWNLOAD_VERSION}/cmake/CPM.cmake
    ${CPM_DOWNLOAD_LOCATION})
endif()

include(${CPM_DOWNLOAD_LOCATION})

function(_set_package_dir_if_exists pkg dir)
    if (NOT DEFINED ENV{NODE_RAPIDS_USE_LOCAL_DEPS_BUILD_DIRS})
        if (EXISTS "${FETCHCONTENT_BASE_DIR}/${dir}-build")
            set(${pkg}_DIR "${FETCHCONTENT_BASE_DIR}/${dir}-build" PARENT_SCOPE)
        endif()
    endif()
endfunction()

# If a target is installed, found by the `find_package` step of CPMFindPackage,
# and marked as IMPORTED, make it globally accessible to consumers of our libs.
function(_fix_cmake_global_defaults target)
    if(TARGET ${target})
        get_target_property(_is_imported ${target} IMPORTED)
        if(_is_imported)
            set_target_properties(${target} PROPERTIES IMPORTED_GLOBAL TRUE)
        endif()
    endif()
endfunction()

if (NOT DEFINED ENV{NODE_RAPIDS_USE_LOCAL_DEPS_BUILD_DIRS})
    execute_process(COMMAND node -p
                    "require('@rapidsai/core').cmake_fetchcontent_base"
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    OUTPUT_VARIABLE NODE_RAPIDS_FETCHCONTENT_BASE_DIR
                    OUTPUT_STRIP_TRAILING_WHITESPACE)

    set(FETCHCONTENT_BASE_DIR "${NODE_RAPIDS_FETCHCONTENT_BASE_DIR}")
    message(STATUS "get_cpm: Using CMake FetchContent base dir: ${FETCHCONTENT_BASE_DIR}")
endif()
