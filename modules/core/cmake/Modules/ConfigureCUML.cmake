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

_get_rapidsai_module_version(cuml VERSION GIT_TAG)

_clean_build_dirs_if_not_fully_built(cuml libcuml++_static)

set(CUVS_LIB cuvs::cuvs_static)

if(NOT TARGET cuml::cuml++_static)
  _get_update_disconnected_state(cuml ${VERSION} UPDATE_DISCONNECTED)

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

  CPMFindPackage(NAME         cuml
    VERSION                 ${VERSION}
    GIT_REPOSITORY          https://github.com/rapidsai/cuml.git
    GIT_TAG                 ${GIT_TAG}
    CUSTOM_CACHE_KEY        ${GIT_TAG}
    GIT_SHALLOW             TRUE
    ${UPDATE_DISCONNECTED}
    SOURCE_SUBDIR           cpp
    OPTIONS                 "CUML_ENABLE_GPU ON"
                            "BUILD_CUML_CPP_LIBRARY ON"
                            "CUML_RAFT_CLONE_ON_PIN ON"
                            "CUML_CUVS_CLONE_ON_PIN ON"
                            "DISABLE_OPENMP ON"
                            "SINGLEGPU ON"
                            "CUDA_STATIC_RUNTIME ON"
                            "CUDA_STATIC_MATH_LIBRARIES ON"
                            "CUML_USE_CUVS_STATIC ON"
                            "CUML_USE_TREELITE_STATIC ON"
                            "BUILD_SHARED_LIBS OFF"
                            "BUILD_CUML_C_LIBRARY OFF"
                            "BUILD_CUML_TESTS OFF"
                            "BUILD_PRIMS_TESTS OFF"
                            "BUILD_CUML_EXAMPLES OFF"
                            "BUILD_CUML_BENCH OFF"
                            "DETECT_CONDA_ENV OFF"
                            "DISABLE_DEPRECATION_WARNINGS ON"
                            "CUDA_WARNINGS_AS_ERRORS OFF"
                            "CUML_COMPILE_DYNAMIC_ONLY OFF"
                            "BUILD_CUML_MG_TESTS OFF"
                            "BUILD_CUML_MPI_COMMS OFF"
                            "CUDA_ENABLE_KERNEL_INFO OFF"
                            "CUDA_ENABLE_LINE_INFO OFF"
                            "NVTX OFF"
                            "USE_CCACHE OFF"
                            "CUML_EXCLUDE_RAFT_FROM_ALL OFF"
                            "CUML_EXCLUDE_TREELITE_FROM_ALL OFF"
                            "CMAKE_C_FLAGS -w"
                            "CMAKE_CXX_FLAGS -w"
                            "CMAKE_CUDA_FLAGS -w"
    PATCHES                 "${CMAKE_CURRENT_LIST_DIR}/../patches/cuml.patch"
  )
endif()

if(NOT cuml_BINARY_DIR OR (NOT cuml_SOURCE_DIR))
  if(NOT DEFINED ENV{NODE_RAPIDS_USE_LOCAL_DEPS_BUILD_DIRS})
    set(cuml_BINARY_DIR "${CPM_BINARY_CACHE}/cuml-build")
    set(cuml_SOURCE_DIR "${CPM_SOURCE_CACHE}/cuml/${GIT_TAG}")
  else()
    set(cuml_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/cuml-build")
    set(cuml_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/cuml/${GIT_TAG}")
  endif()
endif()

rapids_export_find_package_root(BUILD Treelite "${cuml_BINARY_DIR}" EXPORT_SET cuml-exports)

set(CPM_cuml_SOURCE "${cuml_SOURCE_DIR}")

include(${CMAKE_CURRENT_LIST_DIR}/link_utils.cmake)
_statically_link_cuda_toolkit_libs(cuml::cuml++_static)

rapids_export_package(INSTALL cuml ${PROJECT_NAME}-exports)
rapids_export_find_package_root(INSTALL cuml "\${PACKAGE_PREFIX_DIR}/lib/cmake/cuml" EXPORT_SET ${PROJECT_NAME}-exports)
rapids_export_find_package_root(INSTALL Treelite "\${PACKAGE_PREFIX_DIR}/lib/cmake/treelite" EXPORT_SET ${PROJECT_NAME}-exports)
rapids_export_find_package_root(INSTALL GPUTreeShap "\${PACKAGE_PREFIX_DIR}/lib/cmake/gputreeshap" EXPORT_SET ${PROJECT_NAME}-exports)
