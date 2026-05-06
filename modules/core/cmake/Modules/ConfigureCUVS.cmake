#=============================================================================
# Copyright (c) 2026, NVIDIA CORPORATION.
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

_get_rapidsai_module_version(cuvs VERSION GIT_TAG)

_clean_build_dirs_if_not_fully_built(cuvs libcuvs_static)

if(NOT TARGET cuvs::cuvs_static)
  _get_update_disconnected_state(cuvs ${VERSION} UPDATE_DISCONNECTED)

  # For raft sub-build
  set(BUILD_C_LIBRARY OFF)
  set(BUILD_CUVS_BENCH OFF)
  set(BUILD_CAGRA_HNSWLIB OFF)
  set(BUILD_MG_ALGOS OFF)
  set(CUDA_STATIC_RUNTIME ON)
  set(CUDA_STATIC_MATH_LIBRARIES ON)
  set(CUVS_COMPILE_DYNAMIC_ONLY OFF)
  set(CUVS_STATIC_RAPIDS_LIBRARIES ON)
  set(DISABLE_DEPRECATION_WARNINGS ON)

  CPMFindPackage(NAME         cuvs
    VERSION                 ${VERSION}
    GIT_REPOSITORY          https://github.com/rapidsai/cuvs.git
    GIT_TAG                 ${GIT_TAG}
    CUSTOM_CACHE_KEY        ${GIT_TAG}
    GIT_SHALLOW             TRUE
    SOURCE_SUBDIR           cpp
    FIND_PACKAGE_ARGUMENTS  "COMPONENTS cuvs_static"
    ${UPDATE_DISCONNECTED}
    OPTIONS                 "BUILD_TESTS OFF"
                            "BUILD_SHARED_LIBS ON"
                            "BUILD_C_LIBRARY OFF"
                            "BUILD_CUVS_BENCH OFF"
                            "BUILD_CAGRA_HNSWLIB OFF"
                            "BUILD_MG_ALGOS OFF"
                            "CUDA_STATIC_MATH_LIBRARIES ON"
                            "CUDA_STATIC_RUNTIME ON"
                            "CUVS_STATIC_RAPIDS_LIBRARIES ON"
                            "CUVS_COMPILE_DYNAMIC_ONLY OFF"
                            "DETECT_CONDA_ENV OFF"
                            "DISABLE_DEPRECATION_WARNINGS ON"
                            "DISABLE_OPENMP ON"
                            "CMAKE_C_FLAGS -w"
                            "CMAKE_CXX_FLAGS -w"
                            "CMAKE_CUDA_FLAGS -w"
  )
endif()

if(NOT cuvs_BINARY_DIR OR (NOT cuvs_SOURCE_DIR))
  if (NOT DEFINED ENV{NODE_RAPIDS_USE_LOCAL_DEPS_BUILD_DIRS})
    set(cuvs_BINARY_DIR "${CPM_BINARY_CACHE}/cuvs-build")
    set(cuvs_SOURCE_DIR "${CPM_SOURCE_CACHE}/cuvs/${GIT_TAG}")
  else()
    set(cuvs_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/cuvs-build")
    set(cuvs_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/cuvs/${GIT_TAG}")
  endif()
endif()

set(CPM_cuvs_SOURCE "${cuvs_SOURCE_DIR}")

include(${CMAKE_CURRENT_LIST_DIR}/link_utils.cmake)
_statically_link_cuda_toolkit_libs(cuvs::cuvs_static)

rapids_export_package(INSTALL cuvs ${PROJECT_NAME}-exports COMPONENTS cuvs_static GLOBAL_TARGETS cuvs_static)
rapids_export_find_package_root(INSTALL cuvs "\${PACKAGE_PREFIX_DIR}/lib/cmake/cuvs" EXPORT_SET ${PROJECT_NAME}-exports)
