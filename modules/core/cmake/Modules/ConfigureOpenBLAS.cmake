#=============================================================================
# Copyright 2022-2023 NVIDIA Corporation
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

function(find_or_configure_OpenBLAS)

  include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/get_cpm.cmake)
  include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/get_version.cmake)

  set(oneValueArgs VERSION REPOSITORY BRANCH PINNED_TAG EXCLUDE_FROM_ALL)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(INTERFACE64 OFF)
  set(BLA_VENDOR OpenBLAS)
  set(BLA_SIZEOF_INTEGER 4)
  set(BLAS_name "OpenBLAS")
  set(SUFFIX64_UNDERSCORE "")

  # TODO: should we find (or build) 64-bit BLAS?
  if(FALSE AND (CMAKE_SIZEOF_VOID_P EQUAL 8))
    set(INTERFACE64 ON)
    set(BLA_SIZEOF_INTEGER 8)
    set(BLAS_name "OpenBLAS64")
    set(SUFFIX64_UNDERSCORE "_64")
  endif()

  set(BLAS_target "openblas${SUFFIX64_UNDERSCORE}")

  set(FIND_PKG_ARGS      ${PKG_VERSION}
      GLOBAL_TARGETS     ${BLAS_target}
      BUILD_EXPORT_SET   ${PROJECT_NAME}-exports
      INSTALL_EXPORT_SET ${PROJECT_NAME}-exports)

  if(PKG_BRANCH)
    set(PKG_PINNED_TAG "${PKG_BRANCH}")
  endif()

  cmake_policy(GET CMP0048 CMP0048_orig)
  cmake_policy(GET CMP0054 CMP0054_orig)
  set(CMAKE_POLICY_DEFAULT_CMP0048 OLD)
  set(CMAKE_POLICY_DEFAULT_CMP0054 NEW)

  _get_update_disconnected_state(BLAS ${PKG_VERSION} UPDATE_DISCONNECTED)

  rapids_cpm_find(BLAS ${FIND_PKG_ARGS}
      CPM_ARGS
        GIT_REPOSITORY   ${PKG_REPOSITORY}
        GIT_TAG          ${PKG_PINNED_TAG}
        GIT_SHALLOW      TRUE
        EXCLUDE_FROM_ALL ${PKG_EXCLUDE_FROM_ALL}
        OPTIONS "USE_CUDA 1"
                "C_LAPACK ON"
                "USE_THREAD ON"
                "NUM_PARALLEL 32"
                "BUILD_TESTING OFF"
                "BUILD_WITHOUT_CBLAS OFF"
                "BUILD_WITHOUT_LAPACK OFF"
                "INTERFACE64 ${INTERFACE64}"
                "USE_OPENMP ${OpenMP_FOUND}"
                "SUFFIX64_UNDERSCORE ${SUFFIX64_UNDERSCORE}")

  set(CMAKE_POLICY_DEFAULT_CMP0048 ${CMP0048_orig})
  set(CMAKE_POLICY_DEFAULT_CMP0054 ${CMP0054_orig})

  if(BLAS_ADDED AND (TARGET ${BLAS_target}))

    # Ensure we export the name of the actual target, not an alias target
    get_target_property(BLAS_aliased_target ${BLAS_target} ALIASED_TARGET)
    if(TARGET ${BLAS_aliased_target})
      set(BLAS_target ${BLAS_aliased_target})
    endif()
    # Make an BLAS::BLAS alias target
    if(NOT TARGET BLAS::BLAS)
      add_library(BLAS::BLAS ALIAS ${BLAS_target})
    endif()

    # Set build INTERFACE_INCLUDE_DIRECTORIES appropriately
    get_target_property(BLAS_include_dirs ${BLAS_target} INCLUDE_DIRECTORIES)
    target_include_directories(${BLAS_target}
        PUBLIC $<BUILD_INTERFACE:${BLAS_BINARY_DIR}>
               # lapack[e] etc. include paths
               $<BUILD_INTERFACE:${BLAS_include_dirs}>
               # contains openblas_config.h
               $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
               # contains cblas.h and f77blas.h
               $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/generated>
               )

    string(JOIN "\n" code_string
      "if(NOT TARGET BLAS::BLAS)"
      "  add_library(BLAS::BLAS ALIAS ${BLAS_target})"
      "endif()"
    )

    install(EXPORT "${BLAS_name}Targets" FILE ${BLAS_name}Targets.cmake
            NAMESPACE ${BLAS_name} DESTINATION "${BLAS_BINARY_DIR}")

    export(EXPORT "${BLAS_name}Targets" NAMESPACE ${BLAS_name}
           FILE "${BLAS_BINARY_DIR}/${BLAS_name}Targets.cmake")

    # Generate openblas-config.cmake in build dir
    rapids_export(BUILD BLAS
      VERSION ${PKG_VERSION}
      EXPORT_SET "${BLAS_name}Targets"
      GLOBAL_TARGETS ${BLAS_target}
      FINAL_CODE_BLOCK code_string)

    # Do `CPMFindPackage(BLAS)` in build dir
    rapids_export_package(BUILD BLAS ${PROJECT_NAME}-exports
      VERSION ${PKG_VERSION} GLOBAL_TARGETS ${BLAS_target})

    # Tell cmake where it can find the generated blas-config.cmake
    include("${rapids-cmake-dir}/export/find_package_root.cmake")
    rapids_export_find_package_root(BUILD BLAS [=[${CMAKE_CURRENT_LIST_DIR}]=] ${PROJECT_NAME}-exports)
  endif()

  set(BLAS_FOUND TRUE PARENT_SCOPE)
  set(BLAS_ADDED ${BLAS_ADDED} PARENT_SCOPE)
  set(BLAS_BINARY_DIR ${BLAS_BINARY_DIR} PARENT_SCOPE)
  set(BLAS_SOURCE_DIR ${BLAS_SOURCE_DIR} PARENT_SCOPE)

  set(LAPACK_FOUND TRUE PARENT_SCOPE)
  set(LAPACK_ADDED ${BLAS_ADDED} PARENT_SCOPE)
  set(LAPACK_ROOT ${BLAS_BINARY_DIR} PARENT_SCOPE)
  set(LAPACK_BINARY_DIR ${BLAS_BINARY_DIR} PARENT_SCOPE)
  set(LAPACK_SOURCE_DIR ${BLAS_SOURCE_DIR} PARENT_SCOPE)

  set(BLA_VENDOR OpenBLAS)
  set(BLAS_ROOT ${BLAS_BINARY_DIR})
  find_package(BLAS REQUIRED)
  set(BLAS_DIR ${BLAS_DIR} PARENT_SCOPE)
  set(BLAS_FOUND ${BLAS_FOUND} PARENT_SCOPE)
  set(BLAS_VERSION ${BLAS_VERSION} PARENT_SCOPE)
  set(BLAS_LIBRARIES ${BLAS_LIBRARIES} PARENT_SCOPE)
  set(BLAS_LIBRARY ${BLAS_LIBRARY} PARENT_SCOPE)
  set(BLAS_LINKER_FLAGS ${BLAS_LINKER_FLAGS} PARENT_SCOPE)
endfunction()

if(NOT DEFINED OPENBLAS_VERSION)
  # Before v0.3.18, OpenBLAS's throws CMake errors when configuring
  set(OPENBLAS_VERSION "0.3.20")
endif()

if(NOT DEFINED OPENBLAS_BRANCH)
  set(OPENBLAS_BRANCH "")
endif()

if(NOT DEFINED OPENBLAS_TAG)
  set(OPENBLAS_TAG v${OPENBLAS_VERSION})
endif()

if(NOT DEFINED OPENBLAS_REPOSITORY)
  set(OPENBLAS_REPOSITORY https://github.com/xianyi/OpenBLAS.git)
endif()

find_or_configure_OpenBLAS(VERSION          ${OPENBLAS_VERSION}
                           REPOSITORY       ${OPENBLAS_REPOSITORY}
                           BRANCH           ${OPENBLAS_BRANCH}
                           PINNED_TAG       ${OPENBLAS_TAG}
                           EXCLUDE_FROM_ALL ${EXCLUDE_OPENBLAS_FROM_ALL}
)
