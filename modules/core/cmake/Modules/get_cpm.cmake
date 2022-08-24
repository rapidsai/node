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

if(DEFINED CPM_SOURCE_CACHE AND
  (DEFINED ENV{CPM_SOURCE_CACHE}) AND
  (DEFINED CPM_DOWNLOAD_VERSION) AND
  (DEFINED CPM_DOWNLOAD_LOCATION))
    if(DEFINED ENV{NODE_RAPIDS_USE_LOCAL_DEPS_BUILD_DIRS})
        message(VERBOSE "get_cpm: CPM already loaded")
        return()
    endif()
    if(DEFINED CPM_BINARY_CACHE AND
      (DEFINED ENV{CPM_BINARY_CACHE}))
      message(VERBOSE "get_cpm: CPM already loaded")
      return()
  endif()
endif()

if (NOT DEFINED ENV{NODE_RAPIDS_USE_LOCAL_DEPS_BUILD_DIRS})
    execute_process(COMMAND node -p
                    "require('@rapidsai/core').cpm_source_cache_path"
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                    OUTPUT_VARIABLE NODE_RAPIDS_CPM_SOURCE_CACHE
                    OUTPUT_STRIP_TRAILING_WHITESPACE)

    set(CPM_SOURCE_CACHE "${NODE_RAPIDS_CPM_SOURCE_CACHE}")
    set(ENV{CPM_SOURCE_CACHE} "${NODE_RAPIDS_CPM_SOURCE_CACHE}")
    message(VERBOSE "get_cpm: Using CPM source cache: $ENV{CPM_SOURCE_CACHE}")

    execute_process(COMMAND node -p
                    "require('@rapidsai/core').cpm_binary_cache_path"
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                    OUTPUT_VARIABLE NODE_RAPIDS_CPM_BINARY_CACHE
                    OUTPUT_STRIP_TRAILING_WHITESPACE)

    set(CPM_BINARY_CACHE "${NODE_RAPIDS_CPM_BINARY_CACHE}/${CMAKE_BUILD_TYPE}")
    set(ENV{CPM_BINARY_CACHE} "${CPM_BINARY_CACHE}")
    message(VERBOSE "get_cpm: Using CPM BINARY cache: $ENV{CPM_BINARY_CACHE}")

    message(VERBOSE "get_cpm: Using CMake FetchContent base dir: ${CPM_BINARY_CACHE}")
    set(FETCHCONTENT_BASE_DIR "${CPM_BINARY_CACHE}" CACHE STRING "" FORCE)
endif()

file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-22.08/RAPIDS.cmake ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
include(${CMAKE_BINARY_DIR}/RAPIDS.cmake)
include(rapids-export)
include(rapids-cmake)
include(rapids-find)
include(rapids-cpm)

execute_process(COMMAND node -p
                "require('@rapidsai/core').cmake_modules_path"
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                OUTPUT_VARIABLE NODE_RAPIDS_CMAKE_MODULES_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE)

rapids_cpm_init(OVERRIDE "${NODE_RAPIDS_CMAKE_MODULES_PATH}/../versions.json")

function(_set_thrust_dir_if_exists)
    if(Thrust_ROOT)
      message(STATUS "get_cpm: Thrust_ROOT is '${Thrust_ROOT}'")
      return()
    endif()
    if (NOT DEFINED ENV{NODE_RAPIDS_USE_LOCAL_DEPS_BUILD_DIRS})
        file(GLOB _thrust_srcs "${CPM_SOURCE_CACHE}/thrust/*/thrust" LIST_DIRECTORIES TRUE)
        foreach(_thrust_src IN LISTS _thrust_srcs)
            if(_thrust_src AND (EXISTS "${_thrust_src}/cmake"))
                message(STATUS "get_cpm: setting Thrust_ROOT to '${_thrust_src}/cmake'")
                set(Thrust_DIR "${_thrust_src}/cmake" PARENT_SCOPE)
                set(Thrust_ROOT "${_thrust_src}/cmake" PARENT_SCOPE)
                break()
            else()
                if(NOT _thrust_src)
                  set(_thrust_src "thrust/cmake")
                endif()
                message(STATUS "get_cpm: not setting Thrust_ROOT because '${_thrust_src}' does not exist")
            endif()
        endforeach()
    endif()
endfunction()

function(_set_package_dir_if_exists pkg dir)
    if (NOT DEFINED ENV{NODE_RAPIDS_USE_LOCAL_DEPS_BUILD_DIRS})
        set(_build_dir "${CPM_BINARY_CACHE}/${dir}-build")
        if(EXISTS "${_build_dir}")
            message(STATUS "get_cpm: setting ${pkg}_ROOT to '${_build_dir}'")
            set(${pkg}_DIR "${_build_dir}" PARENT_SCOPE)
            set(${pkg}_ROOT "${_build_dir}" PARENT_SCOPE)
        else()
            message(STATUS "get_cpm: not setting ${pkg}_ROOT because '${_build_dir}' does not exist")
        endif()
    endif()
endfunction()

function(_clean_build_dirs_if_not_fully_built dir libname)
    if (NOT DEFINED ENV{NODE_RAPIDS_USE_LOCAL_DEPS_BUILD_DIRS})
        if (EXISTS "${CPM_BINARY_CACHE}/${dir}-build/${libname}.a")
            message(STATUS "get_cpm: not clearing shared build dirs since '${CPM_BINARY_CACHE}/${dir}-build/${libname}.a' exists")
        elseif (EXISTS "${CPM_BINARY_CACHE}/${dir}-build/${libname}.so")
            message(STATUS "get_cpm: not clearing shared build dirs since '${CPM_BINARY_CACHE}/${dir}-build/${libname}.so' exists")
        else()
            file(REMOVE_RECURSE "${CPM_BINARY_CACHE}/${dir}-build")
            file(REMOVE_RECURSE "${CPM_BINARY_CACHE}/${dir}-subbuild")
            message(STATUS "get_cpm: clearing shared build dirs since '${CPM_BINARY_CACHE}/${dir}-build/${libname}.(a|so)' does not exist")
        endif()
    endif()
endfunction()

# If a target is installed, found by the `find_package` step of CPMFindPackage,
# and marked as IMPORTED, make it globally accessible to consumers of our libs.
function(_fix_cmake_global_defaults target)
    if(TARGET ${target})
        get_target_property(_is_imported ${target} IMPORTED)
        get_target_property(_already_global ${target} IMPORTED_GLOBAL)
        if(_is_imported AND NOT _already_global)
            set_target_properties(${target} PROPERTIES IMPORTED_GLOBAL TRUE)
        endif()
        get_target_property(_aliased_target ${target} ALIASED_TARGET)
        if (_aliased_target)
            _fix_cmake_global_defaults(${_aliased_target})
        endif()
    endif()
endfunction()

function(_set_interface_include_dirs_as_system target)
    get_target_property(_real ${target} ALIASED_TARGET)
    if (NOT TARGET ${_real})
        set(_real ${target})
    endif()
    if (TARGET ${_real})
        get_target_property(normal_includes ${target} INTERFACE_INCLUDE_DIRECTORIES)
        get_target_property(system_includes ${target} INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
        if (normal_includes)
            if (NOT system_includes)
                set(system_includes ${normal_includes})
            else()
                list(APPEND system_includes ${normal_includes})
            endif()
            set_property(TARGET ${_real} PROPERTY INTERFACE_INCLUDE_DIRECTORIES "")
            target_include_directories(${_real} SYSTEM INTERFACE ${system_includes})
        endif()
    endif()
endfunction()

function(_get_major_minor_version version out_var)
    if(${version} MATCHES [=[([0-9]+)\.([0-9]+)\.([0-9]+)]=])
        set(${out_var} "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}" PARENT_SCOPE)
    else()
        set(${out_var} "${version}" PARENT_SCOPE)
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
