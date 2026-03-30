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

include(${CMAKE_CURRENT_LIST_DIR}/get_cpm.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/get_version.cmake)

set(cuspatial_VERSION 25.04.00)
set(cuspatial_GIT_TAG branch-25.04)

_get_rapidsai_module_version(cuspatial VERSION GIT_TAG)
set(VERSION 25.04)

_clean_build_dirs_if_not_fully_built(cuspatial libcuspatial)

if(NOT TARGET cuspatial::cuspatial)
  _get_update_disconnected_state(cuspatial ${VERSION} UPDATE_DISCONNECTED)
  CPMFindPackage(NAME         cuspatial
    VERSION                 ${VERSION}
    GIT_REPOSITORY          https://github.com/rapidsai/cuspatial.git
    GIT_TAG                 ${GIT_TAG}
    CUSTOM_CACHE_KEY        ${GIT_TAG}
    GIT_SHALLOW             TRUE
    ${UPDATE_DISCONNECTED}
    SOURCE_SUBDIR           cpp
    OPTIONS                 "BUILD_TESTS OFF"
                            "BUILD_BENCHMARKS OFF"
                            "BUILD_SHARED_LIBS OFF"
                            "CUDA_STATIC_RUNTIME ON"
                            "PER_THREAD_DEFAULT_STREAM ON"
                            "DISABLE_DEPRECATION_WARNING ON"
                            "DISABLE_DEPRECATION_WARNINGS ON"
    PATCHES                 "${CMAKE_CURRENT_LIST_DIR}/../patches/cuspatial.patch"
  )

  if(cuspatial_ADDED)
    rapids_export(
      INSTALL ranger
      VERSION 00.01.00
      EXPORT_SET ranger-exports
      GLOBAL_TARGETS ranger
      NAMESPACE ranger::
    )
  endif()
endif()

if(NOT cuspatial_BINARY_DIR OR (NOT cuspatial_SOURCE_DIR))
  if(NOT DEFINED ENV{NODE_RAPIDS_USE_LOCAL_DEPS_BUILD_DIRS})
    set(cuspatial_BINARY_DIR "${CPM_BINARY_CACHE}/cuspatial-build")
    set(cuspatial_SOURCE_DIR "${CPM_SOURCE_CACHE}/cuspatial/${GIT_TAG}")
  else()
    set(cuspatial_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/cuspatial-build")
    set(cuspatial_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/cuspatial/${GIT_TAG}")
  endif()
endif()

set(CPM_cuspatial_SOURCE "${cuspatial_SOURCE_DIR}")

include(${CMAKE_CURRENT_LIST_DIR}/link_utils.cmake)
_statically_link_cuda_toolkit_libs(cuspatial::cuspatial)

rapids_export_package(INSTALL ranger ${PROJECT_NAME}-exports)
rapids_export_package(INSTALL cuspatial ${PROJECT_NAME}-exports)
rapids_export_find_package_root(INSTALL cuspatial "\${PACKAGE_PREFIX_DIR}/lib/cmake/cuspatial" EXPORT_SET ${PROJECT_NAME}-exports)
rapids_export_find_package_root(INSTALL ranger "\${PACKAGE_PREFIX_DIR}/lib/cmake/ranger" EXPORT_SET ${PROJECT_NAME}-exports)
