#=============================================================================
# Copyright (c) 2020-2026, NVIDIA CORPORATION.
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

_get_rapidsai_module_version(cugraph VERSION GIT_TAG)

_clean_build_dirs_if_not_fully_built(cugraph libcugraph)

if(NOT TARGET cugraph::cugraph)
  _get_update_disconnected_state(cugraph ${VERSION} UPDATE_DISCONNECTED)

  # For raft/cuvs sub-builds
  set(BUILD_C_LIBRARY OFF)
  set(BUILD_CUVS_BENCH OFF)
  set(BUILD_CAGRA_HNSWLIB OFF)
  set(BUILD_MG_ALGOS OFF)
  set(CUDA_STATIC_RUNTIME ON)
  set(CUDA_STATIC_MATH_LIBRARIES ON)
  set(CUVS_COMPILE_DYNAMIC_ONLY OFF)
  set(CUVS_STATIC_RAPIDS_LIBRARIES ON)
  set(DISABLE_DEPRECATION_WARNINGS ON)

  CPMFindPackage(NAME         cugraph
    VERSION                 ${VERSION}
    GIT_REPOSITORY          https://github.com/rapidsai/cugraph.git
    GIT_TAG                 ${GIT_TAG}
    CUSTOM_CACHE_KEY        ${GIT_TAG}
    GIT_SHALLOW             TRUE
    ${UPDATE_DISCONNECTED}
    SOURCE_SUBDIR           cpp
    OPTIONS                 "BUILD_TESTS OFF"
                            "BUILD_BENCHMARKS OFF"
                            "BUILD_SHARED_LIBS OFF"
                            "BUILD_CUGRAPH_MG_TESTS OFF"
                            "CUDA_STATIC_RUNTIME ON"
                            "USE_RAFT_STATIC ON"
                            "CUGRAPH_COMPILE_CUVS ON"
                            "CUGRAPH_USE_CUVS_STATIC ON"
                            "BUILD_CUGRAPH_MG_TESTS OFF"
    PATCHES                 "${CMAKE_CURRENT_LIST_DIR}/../patches/cugraph.patch"
  )
endif()

if(NOT cugraph_BINARY_DIR OR (NOT cugraph_SOURCE_DIR))
  if(NOT DEFINED ENV{NODE_RAPIDS_USE_LOCAL_DEPS_BUILD_DIRS})
    set(cugraph_BINARY_DIR "${CPM_BINARY_CACHE}/cugraph-build")
    set(cugraph_SOURCE_DIR "${CPM_SOURCE_CACHE}/cugraph/${GIT_TAG}")
  else()
    set(cugraph_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/cugraph-build")
    set(cugraph_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/cugraph/${GIT_TAG}")
  endif()
endif()

set(CPM_cugraph_SOURCE "${cugraph_SOURCE_DIR}")

include(${CMAKE_CURRENT_LIST_DIR}/link_utils.cmake)
_statically_link_cuda_toolkit_libs(cugraph::cugraph)

rapids_export_package(INSTALL cugraph ${PROJECT_NAME}-exports)
rapids_export_find_package_root(INSTALL cugraph "\${PACKAGE_PREFIX_DIR}/lib/cmake/cugraph" EXPORT_SET ${PROJECT_NAME}-exports)
