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

_get_rapidsai_module_version(rmm VERSION GIT_TAG)

if(NOT TARGET rmm::rmm)
  _get_update_disconnected_state(rmm ${VERSION} UPDATE_DISCONNECTED)
  CPMFindPackage(NAME        rmm
      VERSION                ${VERSION}
      GIT_REPOSITORY         https://github.com/rapidsai/rmm.git
      GIT_TAG                ${GIT_TAG}
      CUSTOM_CACHE_KEY       ${GIT_TAG}
      GIT_SHALLOW            TRUE
      SOURCE_SUBDIR          cpp
      ${UPDATE_DISCONNECTED}
      OPTIONS                "BUILD_TESTS OFF"
                             "BUILD_BENCHMARKS OFF"
                             "BUILD_SHARED_LIBS OFF"
                             "RMM_LOGGING_LEVEL TRACE"
                             "DISABLE_DEPRECATION_WARNINGS ${DISABLE_DEPRECATION_WARNINGS}")
endif()

if(NOT rmm_BINARY_DIR OR (NOT rmm_SOURCE_DIR))
  if(NOT DEFINED ENV{NODE_RAPIDS_USE_LOCAL_DEPS_BUILD_DIRS})
    set(rmm_BINARY_DIR "${CPM_BINARY_CACHE}/rmm-build")
    set(rmm_SOURCE_DIR "${CPM_SOURCE_CACHE}/rmm/${GIT_TAG}")
  else()
    set(rmm_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/rmm-build")
    set(rmm_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/rmm/${GIT_TAG}")
  endif()
endif()

set(CPM_rmm_SOURCE "${rmm_SOURCE_DIR}")
set(CPM_rapids_logger_SOURCE "${rapids_logger_SOURCE_DIR}")

include(${CMAKE_CURRENT_LIST_DIR}/link_utils.cmake)
_statically_link_cuda_toolkit_libs(rmm::rmm)
_statically_link_cuda_toolkit_libs(rapids_logger::rapids_logger)

rapids_export_package(INSTALL rmm ${PROJECT_NAME}-exports)
rapids_export_find_package_root(INSTALL rmm "\${PACKAGE_PREFIX_DIR}/lib/cmake/rmm" EXPORT_SET ${PROJECT_NAME}-exports)
rapids_export_find_package_root(INSTALL fmt "\${PACKAGE_PREFIX_DIR}/lib/cmake/fmt" EXPORT_SET ${PROJECT_NAME}-exports)
rapids_export_find_package_root(INSTALL nvtx3 "\${PACKAGE_PREFIX_DIR}/lib/cmake/nvtx3" EXPORT_SET ${PROJECT_NAME}-exports)
rapids_export_find_package_root(INSTALL spdlog "\${PACKAGE_PREFIX_DIR}/lib/cmake/spdlog" EXPORT_SET ${PROJECT_NAME}-exports)
rapids_export_find_package_root(INSTALL CCCL "\${PACKAGE_PREFIX_DIR}/lib/rapids/cmake/cccl" EXPORT_SET ${PROJECT_NAME}-exports)
rapids_export_find_package_root(INSTALL rapids_logger "\${PACKAGE_PREFIX_DIR}/lib/cmake/rapids_logger" EXPORT_SET ${PROJECT_NAME}-exports)
