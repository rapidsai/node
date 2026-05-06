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

include(${CMAKE_CURRENT_LIST_DIR}/get_cpm.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/get_version.cmake)

_get_rapidsai_module_version(raft VERSION GIT_TAG)

_clean_build_dirs_if_not_fully_built(raft libraft)

if(NOT TARGET raft::compiled_static)
  _get_update_disconnected_state(raft ${VERSION} UPDATE_DISCONNECTED)

  set(BUILD_C_LIBRARY OFF)
  set(BUILD_CUVS_BENCH OFF)
  set(BUILD_CAGRA_HNSWLIB OFF)
  set(BUILD_MG_ALGOS OFF)
  set(CUDA_STATIC_RUNTIME ON)
  set(CUDA_STATIC_MATH_LIBRARIES ON)
  set(CUVS_COMPILE_DYNAMIC_ONLY OFF)
  set(CUVS_STATIC_RAPIDS_LIBRARIES ON)
  set(DISABLE_DEPRECATION_WARNINGS ON)

  CPMFindPackage(NAME         raft
    VERSION                 ${VERSION}
    GIT_REPOSITORY          https://github.com/rapidsai/raft.git
    GIT_TAG                 ${GIT_TAG}
    CUSTOM_CACHE_KEY        ${GIT_TAG}
    GIT_SHALLOW             TRUE
    SOURCE_SUBDIR           cpp
    FIND_PACKAGE_ARGUMENTS  "COMPONENTS compiled compiled-static"
    ${UPDATE_DISCONNECTED}
    OPTIONS                 "BUILD_TESTS OFF"
                            # "BLA_VENDOR OpenBLAS"
                            "BUILD_SHARED_LIBS OFF"
                            "BUILD_PRIMS_BENCH OFF"
                            "BUILD_CAGRA_HNSWLIB OFF"
                            "CUDA_STATIC_MATH_LIBRARIES ON"
                            "CUDA_STATIC_RUNTIME ON"
                            "RAFT_COMPILE_LIBRARY ON"
                            "RAFT_COMPILE_DYNAMIC_ONLY OFF"
                            "DISABLE_DEPRECATION_WARNINGS ON"
                            "DISABLE_OPENMP ON"
                            "CMAKE_C_FLAGS -w"
                            "CMAKE_CXX_FLAGS -w"
                            "CMAKE_CUDA_FLAGS -w"
    PATCHES                 "${CMAKE_CURRENT_LIST_DIR}/../patches/raft.patch"
  )
endif()

if(NOT raft_BINARY_DIR OR (NOT raft_SOURCE_DIR))
  if(NOT DEFINED ENV{NODE_RAPIDS_USE_LOCAL_DEPS_BUILD_DIRS})
    set(raft_BINARY_DIR "${CPM_BINARY_CACHE}/raft-build")
    set(raft_SOURCE_DIR "${CPM_SOURCE_CACHE}/raft/${GIT_TAG}")
  else()
    set(raft_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/raft-build")
    set(raft_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/raft/${GIT_TAG}")
  endif()
endif()

set(CPM_raft_SOURCE "${raft_SOURCE_DIR}")

include(${CMAKE_CURRENT_LIST_DIR}/link_utils.cmake)
_statically_link_cuda_toolkit_libs(raft::raft)
_statically_link_cuda_toolkit_libs(raft::compiled_static)
_statically_link_cuda_toolkit_libs(raft::raft_lib_static)

rapids_export_package(INSTALL raft ${PROJECT_NAME}-exports COMPONENTS compiled-static GLOBAL_TARGETS compiled_static)
rapids_export_find_package_root(INSTALL raft "\${PACKAGE_PREFIX_DIR}/lib/cmake/raft" EXPORT_SET ${PROJECT_NAME}-exports)
