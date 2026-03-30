# =============================================================================
# Copyright (c) 2020-2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================
include_guard(GLOBAL)

# Finding arrow is far more complex than it should be, and as a result we violate multiple linting
# rules aiming to limit complexity. Since all our other CMake scripts conform to expectations
# without undue difficulty, disabling those rules for just this function is our best approach for
# now. The spacing between this comment, the cmake-lint directives, and the function docstring is
# necessary to prevent cmake-format from trying to combine the lines.

# cmake-lint: disable=R0912,R0913,R0915

include(${CMAKE_CURRENT_LIST_DIR}/get_cpm.cmake)

string(TOLOWER "${CMAKE_BUILD_TYPE}" _build_type)
_clean_build_dirs_if_not_fully_built(arrow "${_build_type}/libarrow")
_clean_build_dirs_if_not_fully_built(arrow "${_build_type}/libarrow_cuda")

if(NOT ARROW_ARMV8_ARCH)
  set(ARROW_ARMV8_ARCH "armv8-a")
endif()

# Set this so Arrow correctly finds the CUDA toolkit when the build machine does not have the CUDA
# driver installed. This must be an env var.
set(ENV{CUDA_LIB_PATH} "${CUDAToolkit_LIBRARY_DIR}/stubs")

# Set this so Arrow doesn't add `-Werror` to
# CMAKE_CXX_FLAGS when CMAKE_BUILD_TYPE=Debug
set(BUILD_WARNING_LEVEL "PRODUCTION")
# set(BUILD_WARNING_LEVEL "PRODUCTION" PARENT_SCOPE)
set(BUILD_WARNING_LEVEL "PRODUCTION" CACHE STRING "" FORCE)

set(Arrow_FOUND OFF)

if(Arrow_DIR)
  # Set this to enable `find_package(ArrowCUDA)`
  set(ArrowCUDA_DIR "${Arrow_DIR}")
  find_package(Arrow REQUIRED QUIET)
  find_package(ArrowCUDA REQUIRED QUIET)
endif()

if(NOT TARGET Arrow::arrow_static)
  _get_update_disconnected_state(Arrow 21.0.0 UPDATE_DISCONNECTED)

  find_package(OpenSSL REQUIRED)

  rapids_cpm_find(Arrow 21.0.0
    GLOBAL_TARGETS      arrow_static arrow_cuda_static
    CPM_ARGS
      GIT_REPOSITORY   https://github.com/apache/arrow.git
      GIT_TAG          apache-arrow-21.0.0
      CUSTOM_CACHE_KEY apache-arrow-21.0.0
      GIT_SHALLOW      TRUE
      SOURCE_SUBDIR    cpp
      ${UPDATE_DISCONNECTED}
      OPTIONS          "CMAKE_VERBOSE_MAKEFILE ON"
                       "CUDA_USE_STATIC_CUDA_RUNTIME ON"
                       "ARROW_IPC ON"
                       "ARROW_CUDA ON"
                       "ARROW_WITH_BACKTRACE ON"
                       "ARROW_FILESYSTEM ON"
                       "ARROW_ACERO OFF"
                       "ARROW_COMPUTE OFF"
                       "ARROW_DATASET OFF"
                       "ARROW_CXXFLAGS -w"
                       "ARROW_JEMALLOC OFF"
                       "ARROW_S3 OFF"
                       "ARROW_ORC OFF"
                       "ARROW_PARQUET OFF"
                       # Arrow modifies CMake's GLOBAL RULE_LAUNCH_COMPILE unless this is off
                       "ARROW_USE_CCACHE OFF"
                       "ARROW_POSITION_INDEPENDENT_CODE ON"
                       "ARROW_ARMV8_ARCH ${ARROW_ARMV8_ARCH}"
                       "ARROW_SIMD_LEVEL NONE"
                       "ARROW_BUILD_STATIC ON"
                       "ARROW_BUILD_SHARED OFF"
                       "ARROW_DEPENDENCY_USE_SHARED OFF"
                       "ARROW_BOOST_USE_SHARED OFF"
                       "ARROW_BROTLI_USE_SHARED OFF"
                       "ARROW_GFLAGS_USE_SHARED OFF"
                       "ARROW_GRPC_USE_SHARED OFF"
                       "ARROW_PROTOBUF_USE_SHARED OFF"
                       "ARROW_ZSTD_USE_SHARED OFF"
                       "xsimd_SOURCE AUTO"
  )
endif()

# Arrow_ADDED: set if CPM downloaded Arrow from Github
# Arrow_FOUND: set if find_package found Arrow on the system/conda/etc.

if(Arrow_ADDED)

  set(ARROW_LIBRARIES "")

  if(TARGET arrow_static)
    list(APPEND ARROW_LIBRARIES arrow_static)
  endif()
  if(TARGET arrow_cuda_static)
    list(APPEND ARROW_LIBRARIES arrow_cuda_static)
  endif()

  # Arrow populates INTERFACE_INCLUDE_DIRECTORIES for the `arrow_static` and `arrow_cuda_static`
  # targets in FindArrow and FindArrowCUDA respectively, so for static source-builds, we have to
  # do it after-the-fact.
  #
  # This only works because we know exactly which components we're using. Don't forget to update
  # this list if we add more!
  #
  foreach(ARROW_LIBRARY ${ARROW_LIBRARIES})
    target_include_directories(
      ${ARROW_LIBRARY}
      INTERFACE "$<BUILD_INTERFACE:${Arrow_BINARY_DIR}/src>"
                "$<BUILD_INTERFACE:${Arrow_SOURCE_DIR}/cpp/src>"
                "$<BUILD_INTERFACE:${Arrow_SOURCE_DIR}/cpp/src/generated>"
                "$<BUILD_INTERFACE:${Arrow_SOURCE_DIR}/cpp/thirdparty/hadoop/include>"
                "$<BUILD_INTERFACE:${Arrow_SOURCE_DIR}/cpp/thirdparty/flatbuffers/include>"
    )
  endforeach()

  # Fix for Arrow static library CMake export errors
  # The `arrow_static` library is leaking a dependency on the object libraries it was built with
  # we need to remove this from the interface, since keeping them around would cause duplicate
  # symbols and CMake export errors
  if(TARGET arrow_static)
    get_target_property(interface_libs arrow_static INTERFACE_LINK_LIBRARIES)
    string(REPLACE "BUILD_INTERFACE:" "BUILD_LOCAL_INTERFACE:" interface_libs "${interface_libs}")
    set_target_properties(arrow_static PROPERTIES INTERFACE_LINK_LIBRARIES "${interface_libs}")
  endif()

  set(arrow_code_string [=[
  if (TARGET Arrow::arrow_static AND (NOT TARGET arrow_static))
      add_library(arrow_static ALIAS Arrow::arrow_static)
  endif()
  if (TARGET arrow_static AND (NOT TARGET Arrow::arrow_static))
      add_library(Arrow::arrow_static ALIAS arrow_static)
  endif()
  if (NOT TARGET arrow::flatbuffers)
      add_library(arrow::flatbuffers INTERFACE IMPORTED)
  endif()
  if (NOT TARGET arrow::hadoop)
      add_library(arrow::hadoop INTERFACE IMPORTED)
  endif()
]=]
  )

  if(NOT TARGET xsimd)
    string(
      APPEND
      arrow_code_string
      "
        if(NOT TARGET xsimd)
          add_library(xsimd INTERFACE IMPORTED)
          target_include_directories(xsimd INTERFACE \"${Arrow_BINARY_DIR}/xsimd_ep/src/xsimd_ep-install/include\")
        endif()
      "
    )
  endif()

  set(PROJECT_BINARY_DIR_prev "${PROJECT_BINARY_DIR}")
  set(PROJECT_BINARY_DIR "${Arrow_BINARY_DIR}")

  rapids_export(
    BUILD Arrow
    VERSION 21.0.0
    EXPORT_SET arrow_targets
    GLOBAL_TARGETS arrow_static
    NAMESPACE Arrow::
    FINAL_CODE_BLOCK arrow_code_string
  )

  set(arrow_cuda_code_string [=[
  if (TARGET ArrowCUDA::arrow_cuda_static AND (NOT TARGET arrow_cuda_static))
      add_library(arrow_cuda_static ALIAS ArrowCUDA::arrow_cuda_static)
  endif()
  if (TARGET arrow_cuda_static AND (NOT TARGET ArrowCUDA::arrow_cuda_static))
      add_library(ArrowCUDA::arrow_cuda_static ALIAS arrow_cuda_static)
  endif()
]=]
  )

  rapids_export(
    BUILD ArrowCUDA
    VERSION 21.0.0
    EXPORT_SET arrow_cuda_targets
    GLOBAL_TARGETS arrow_cuda_static
    NAMESPACE ArrowCUDA::
    FINAL_CODE_BLOCK arrow_cuda_code_string
  )

  set(PROJECT_BINARY_DIR "${PROJECT_BINARY_DIR_prev}")

  # We generate the arrow-config and arrowcuda-config files when we built arrow locally, so always
  # do `find_dependency`
  rapids_export_package(BUILD Arrow ${PROJECT_NAME}-exports)

  # We have to generate the find_dependency(ArrowCUDA) ourselves since we need to specify
  # ArrowCUDA_DIR to be where Arrow was found, since Arrow packages ArrowCUDA.config in a
  # non-standard location
  rapids_export_package(BUILD ArrowCUDA ${PROJECT_NAME}-exports)

  include("${rapids-cmake-dir}/export/find_package_root.cmake")
  rapids_export_find_package_root(BUILD Arrow "${Arrow_BINARY_DIR}" ${PROJECT_NAME}-exports)
  rapids_export_find_package_root(BUILD ArrowCUDA "${Arrow_BINARY_DIR}" ${PROJECT_NAME}-exports)

  if(NOT ("${arrow_code_string}" STREQUAL ""))
    cmake_language(EVAL CODE "${arrow_code_string}")
  endif()
  if(NOT ("${arrow_cuda_code_string}" STREQUAL ""))
    cmake_language(EVAL CODE "${arrow_cuda_code_string}")
  endif()
endif()

# set(Arrow_ADDED ${Arrow_ADDED} PARENT_SCOPE)
# set(Arrow_FOUND ${Arrow_FOUND} PARENT_SCOPE)
# set(ArrowCUDA_FOUND ${ArrowCUDA_FOUND} PARENT_SCOPE)
# set(ARROW_LIBRARIES ${ARROW_LIBRARIES} PARENT_SCOPE)
# set(Arrow_DIR "${Arrow_DIR}" PARENT_SCOPE)
# set(ArrowCUDA_DIR "${ArrowCUDA_DIR}" PARENT_SCOPE)
# set(Arrow_BINARY_DIR "${Arrow_BINARY_DIR}" PARENT_SCOPE)
# set(Arrow_SOURCE_DIR "${Arrow_SOURCE_DIR}" PARENT_SCOPE)
# set(Arrow_VERSION "${Arrow_VERSION}" PARENT_SCOPE)
# set(CPM_Arrow_SOURCE "${Arrow_SOURCE_DIR}" PARENT_SCOPE)

if(NOT (Arrow_BINARY_DIR OR Arrow_SOURCE_DIR))
  if(NOT DEFINED ENV{NODE_RAPIDS_USE_LOCAL_DEPS_BUILD_DIRS})
    set(Arrow_BINARY_DIR "${CPM_BINARY_CACHE}/arrow-build")
    set(Arrow_SOURCE_DIR "${CPM_SOURCE_CACHE}/arrow/${GIT_TAG}")
  else()
    set(Arrow_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/arrow-build")
    set(Arrow_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/arrow/${GIT_TAG}")
  endif()
endif()

set(CPM_Arrow_SOURCE "${Arrow_SOURCE_DIR}")

if(NOT (ArrowCUDA_BINARY_DIR OR ArrowCUDA_SOURCE_DIR))
  if(NOT DEFINED ENV{NODE_RAPIDS_USE_LOCAL_DEPS_BUILD_DIRS})
    set(ArrowCUDA_BINARY_DIR "${CPM_BINARY_CACHE}/arrow-build")
    set(ArrowCUDA_SOURCE_DIR "${CPM_SOURCE_CACHE}/arrow/${GIT_TAG}")
  else()
    set(ArrowCUDA_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/arrow-build")
    set(ArrowCUDA_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/arrow/${GIT_TAG}")
  endif()
endif()

set(CPM_ArrowCUDA_SOURCE "${ArrowCUDA_SOURCE_DIR}")

include(${CMAKE_CURRENT_LIST_DIR}/link_utils.cmake)
_statically_link_cuda_toolkit_libs(ArrowCUDA::arrow_cuda_static)

rapids_export_package(INSTALL Arrow ${PROJECT_NAME}-exports)
rapids_export_package(INSTALL ArrowCUDA ${PROJECT_NAME}-exports)

include("${rapids-cmake-dir}/export/find_package_root.cmake")
rapids_export_find_package_root(INSTALL Arrow "\${PACKAGE_PREFIX_DIR}/lib/cmake/Arrow" EXPORT_SET ${PROJECT_NAME}-exports)
rapids_export_find_package_root(INSTALL ArrowCUDA "\${PACKAGE_PREFIX_DIR}/lib/cmake/ArrowCUDA" EXPORT_SET ${PROJECT_NAME}-exports)

if(NOT Arrow_ADDED)
  if(NOT TARGET Arrow::arrow_static)
    find_package(Arrow REQUIRED)
  endif()

  if(NOT TARGET ArrowCUDA::arrow_cuda_static)
    find_package(ArrowCUDA REQUIRED)
  endif()
endif()
