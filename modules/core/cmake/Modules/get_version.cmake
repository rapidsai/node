#=============================================================================
# Copyright (c) 2022-2026, NVIDIA CORPORATION.
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

function(_get_major_minor_version version out_var)
    if(${version} MATCHES [=[([0-9]+)\.([0-9]+)\.([0-9]+)]=])
        set(${out_var} "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}" PARENT_SCOPE)
    else()
        set(${out_var} "${version}" PARENT_SCOPE)
    endif()
endfunction()

function(_get_rapidsai_module_version pkg out_ver_ out_tag_)
  set(ver_ "26.04.00")

  if(DEFINED ${pkg}_VERSION)
    set(ver_ "${${pkg}_VERSION}")
  elseif(DEFINED RAPIDS_VERSION)
    set(ver_ "${RAPIDS_VERSION}")
  elseif(DEFINED ENV{RAPIDS_VERSION})
    set(ver_ "$ENV{RAPIDS_VERSION}")
  endif()

  if(DEFINED ${pkg}_GIT_TAG)
    set(tag_ "${${pkg}_GIT_TAG}")
  else()
    _get_major_minor_version(${ver_} tag_)
    set(tag_ "release/${tag_}")
  endif()

  file(
    DOWNLOAD "https://raw.githubusercontent.com/rapidsai/${pkg}/${tag_}/VERSION"
    STATUS ${pkg}_download_versioned_result
  )

  list(POP_FRONT ${pkg}_download_versioned_result ${pkg}_download_versioned_status)

  if(NOT "${${pkg}_download_versioned_status}" STREQUAL "0")
    list(POP_FRONT ${pkg}_download_versioned_result ${pkg}_download_versioned_message)
    message(STATUS "Failed to download ${pkg}@${tag_}/VERSION:\n HTTP STATUS ${${pkg}_download_versioned_status}: ${${pkg}_download_versioned_message}")
    message(STATUS "Trying ${pkg}@main")

    set(tag_ main)

    file(
      DOWNLOAD "https://raw.githubusercontent.com/rapidsai/${pkg}/${tag_}/VERSION"
      STATUS ${pkg}_download_versioned_result
    )

    list(POP_FRONT ${pkg}_download_versioned_result ${pkg}_download_versioned_status)

    if(NOT "${${pkg}_download_versioned_status}" STREQUAL "0")
      list(POP_FRONT ${pkg}_download_versioned_result ${pkg}_download_versioned_message)
      message(FATAL_ERROR "Failed to download download ${pkg}@${tag_}/VERSION:\n${${pkg}_download_versioned_message}")
    endif()
  endif()


  set(RAPIDS_VERSION "${ver_}" PARENT_SCOPE)
  set(${out_ver_} "${ver_}" PARENT_SCOPE)
  set(${out_tag_} "${tag_}" PARENT_SCOPE)
endfunction()
