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

_get_rapidsai_module_version(cudf VERSION GIT_TAG)

_clean_build_dirs_if_not_fully_built(cudf libcudf)

if(NOT TARGET cudf::cudf)
  _get_update_disconnected_state(cudf ${VERSION} UPDATE_DISCONNECTED)
  CPMFindPackage(NAME     cudf
    VERSION             ${VERSION}
    GIT_REPOSITORY      https://github.com/rapidsai/cudf.git
    GIT_TAG             ${GIT_TAG}
    CUSTOM_CACHE_KEY    ${GIT_TAG}
    GIT_SHALLOW         TRUE
    ${UPDATE_DISCONNECTED}
    SOURCE_SUBDIR       cpp
    OPTIONS             "BUILD_TESTS OFF"
                        "BUILD_BENCHMARKS OFF"
                        "BUILD_SHARED_LIBS OFF"
                        "JITIFY_USE_CACHE ON"
                        "CUDA_STATIC_RUNTIME ON"
                        "CUDA_WARNINGS_AS_ERRORS OFF"
                        "CUDF_USE_ARROW_STATIC ON"
                        "CUDF_USE_PROPRIETARY_NVCOMP ON"
                        "CUDF_USE_PER_THREAD_DEFAULT_STREAM ON"
                        "DISABLE_DEPRECATION_WARNINGS ON"
    PATCHES             "${CMAKE_CURRENT_LIST_DIR}/../patches/cudf.patch"
)
endif()

if(NOT cudf_BINARY_DIR OR (NOT cudf_SOURCE_DIR))
  if(NOT DEFINED ENV{NODE_RAPIDS_USE_LOCAL_DEPS_BUILD_DIRS})
    set(cudf_BINARY_DIR "${CPM_BINARY_CACHE}/cudf-build")
    set(cudf_SOURCE_DIR "${CPM_SOURCE_CACHE}/cudf/${GIT_TAG}")
  else()
    set(cudf_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/cudf-build")
    set(cudf_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/cudf/${GIT_TAG}")
  endif()
endif()

set(CPM_cudf_SOURCE "${cudf_SOURCE_DIR}")

include(${CMAKE_CURRENT_LIST_DIR}/link_utils.cmake)
_statically_link_cuda_toolkit_libs(cudf::cudf)
_statically_link_cuda_toolkit_libs(cudf::cudftestutil)

set(_libname)

foreach(_libname IN ITEMS bs_thread_pool cuco cudf CURL flatbuffers nanoarrow nvcomp roaring zstd)
  rapids_export_package(INSTALL ${_libname} ${PROJECT_NAME}-exports)
  rapids_export_find_package_root(INSTALL ${_libname} "\${PACKAGE_PREFIX_DIR}/lib/cmake/${_libname}" EXPORT_SET ${PROJECT_NAME}-exports)
endforeach()

rapids_export_package(INSTALL KvikIO ${PROJECT_NAME}-exports)
rapids_export_find_package_root(INSTALL KvikIO "\${PACKAGE_PREFIX_DIR}/lib/cmake/kvikio" EXPORT_SET ${PROJECT_NAME}-exports)

unset(_libname)
