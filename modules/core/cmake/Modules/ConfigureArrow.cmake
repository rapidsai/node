# =============================================================================
# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

# This function finds arrow and sets any additional necessary environment variables.
function(find_and_configure_arrow VERSION BUILD_STATIC ENABLE_S3 ENABLE_ORC ENABLE_PYTHON
         ENABLE_PARQUET
)
  include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/get_cpm.cmake)

  string(TOLOWER "${CMAKE_BUILD_TYPE}" _build_type)
  _clean_build_dirs_if_not_fully_built(arrow "${_build_type}/libarrow")
  _clean_build_dirs_if_not_fully_built(arrow "${_build_type}/libparquet")
  _clean_build_dirs_if_not_fully_built(arrow "${_build_type}/libarrow_cuda")
  _clean_build_dirs_if_not_fully_built(arrow "${_build_type}/libarrow_dataset")

  _set_package_dir_if_exists(Arrow arrow)
  _set_package_dir_if_exists(Parquet arrow)
  _set_package_dir_if_exists(ArrowCUDA arrow)
  _set_package_dir_if_exists(ArrowDataset arrow)

  # if (NOT DEFINED ENV{NODE_RAPIDS_USE_LOCAL_DEPS_BUILD_DIRS})
  #   set(_pkg_names Arrow Parquet ArrowCUDA ArrowDataset)
  #   set(_tgt_names arrow_static parquet_static arrow_cuda_static arrow_dataset_static)
  #   set(_lib_names "libarrow.a" "libparquet.a" "libarrow_cuda.a" "libarrow_dataset.a")
  #   set(_build_dirs
  #     "${CPM_BINARY_CACHE}/arrow-build/src/arrow"
  #     "${CPM_BINARY_CACHE}/arrow-build/src/parquet"
  #     "${CPM_BINARY_CACHE}/arrow-build/src/arrow/gpu"
  #     "${CPM_BINARY_CACHE}/arrow-build/src/arrow/dataset"
  #   )
  #   string(TOLOWER "${CMAKE_BUILD_TYPE}" _build_type)
  #   foreach(_elem IN ZIP_LISTS _pkg_names _build_dirs _tgt_names _lib_names)
  #     set(_pkg ${_elem_0})
  #     set(_dir ${_elem_1})
  #     set(_tgt_name ${_elem_2})
  #     set(_lib_name ${_elem_3})
  #     if(EXISTS "${_dir}")
  #       set(${_pkg}_DIR "${CPM_BINARY_CACHE}/arrow-build")
  #       set(${_pkg}_ROOT "${CPM_BINARY_CACHE}/arrow-build")
  #       message(STATUS "get_cpm: setting ${_pkg}_ROOT to '${CPM_BINARY_CACHE}/arrow-build'")
  #     else()
  #         message(STATUS "get_cpm: not setting ${_pkg}_ROOT because '${_dir}' does not exist")
  #     endif()
  #   endforeach()
  # endif()

  set(ARROW_BUILD_SHARED ON)
  set(ARROW_BUILD_STATIC OFF)

  if(NOT ARROW_ARMV8_ARCH)
    set(ARROW_ARMV8_ARCH "armv8-a")
  endif()

  if(NOT ARROW_SIMD_LEVEL)
    set(ARROW_SIMD_LEVEL "NONE")
  endif()

  if(BUILD_STATIC)
    set(ARROW_BUILD_STATIC ON)
    set(ARROW_BUILD_SHARED OFF)
    # Turn off CPM using `find_package` so we always download and make sure we get proper static
    # library
    # set(CPM_DOWNLOAD_ALL TRUE)
  endif()

  set(ARROW_PYTHON_OPTIONS "")
  if(ENABLE_PYTHON)
    list(APPEND ARROW_PYTHON_OPTIONS "ARROW_PYTHON ON")
    # Arrow's logic to build Boost from source is busted, so we have to get it from the system.
    list(APPEND ARROW_PYTHON_OPTIONS "BOOST_SOURCE SYSTEM")
    list(APPEND ARROW_PYTHON_OPTIONS "ARROW_DEPENDENCY_SOURCE AUTO")
  endif()

  set(ARROW_PARQUET_OPTIONS "")
  if(ENABLE_PARQUET)
    # Arrow's logic to build Boost from source is busted, so we have to get it from the system.
    list(APPEND ARROW_PARQUET_OPTIONS "BOOST_SOURCE SYSTEM")
    list(APPEND ARROW_PARQUET_OPTIONS "Thrift_SOURCE BUNDLED")
    list(APPEND ARROW_PARQUET_OPTIONS "ARROW_DEPENDENCY_SOURCE AUTO")
  endif()

  # Set this so Arrow correctly finds the CUDA toolkit when the build machine does not have the CUDA
  # driver installed. This must be an env var.
  set(ENV{CUDA_LIB_PATH} "${CUDAToolkit_LIBRARY_DIR}/stubs")

  include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/get_cpm.cmake)
  include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/get_version.cmake)

  # Set this so Arrow doesn't add `-Werror` to
  # CMAKE_CXX_FLAGS when CMAKE_BUILD_TYPE=Debug
  set(BUILD_WARNING_LEVEL "PRODUCTION")
  set(BUILD_WARNING_LEVEL "PRODUCTION" PARENT_SCOPE)
  set(BUILD_WARNING_LEVEL "PRODUCTION" CACHE STRING "" FORCE)

  rapids_cpm_find(
    Arrow ${VERSION}
    GLOBAL_TARGETS arrow_shared arrow_static
                   parquet_shared parquet_static
                   arrow_cuda_shared arrow_cuda_static
                   arrow_dataset_shared arrow_dataset_static
    CPM_ARGS
    GIT_REPOSITORY https://github.com/apache/arrow.git
    GIT_TAG apache-arrow-${VERSION}
    GIT_SHALLOW TRUE SOURCE_SUBDIR cpp
    OPTIONS "CMAKE_VERBOSE_MAKEFILE ON"
            "CUDA_USE_STATIC_CUDA_RUNTIME OFF"
            "ARROW_IPC ON"
            "ARROW_CUDA ON"
            "ARROW_DATASET ON"
            "ARROW_WITH_BACKTRACE ON"
            "ARROW_CXXFLAGS -w"
            "ARROW_JEMALLOC OFF"
            "ARROW_S3 ${ENABLE_S3}"
            "ARROW_ORC ${ENABLE_ORC}"
            # e.g. needed by blazingsql-io
            ${ARROW_PARQUET_OPTIONS}
            "ARROW_PARQUET ${ENABLE_PARQUET}"
            ${ARROW_PYTHON_OPTIONS}
            # Arrow modifies CMake's GLOBAL RULE_LAUNCH_COMPILE unless this is off
            "ARROW_USE_CCACHE OFF"
            "ARROW_POSITION_INDEPENDENT_CODE ON"
            "ARROW_ARMV8_ARCH ${ARROW_ARMV8_ARCH}"
            "ARROW_SIMD_LEVEL ${ARROW_SIMD_LEVEL}"
            "ARROW_BUILD_STATIC ${ARROW_BUILD_STATIC}"
            "ARROW_BUILD_SHARED ${ARROW_BUILD_SHARED}"
            "ARROW_DEPENDENCY_USE_SHARED ${ARROW_BUILD_SHARED}"
            "ARROW_BOOST_USE_SHARED ${ARROW_BUILD_SHARED}"
            "ARROW_BROTLI_USE_SHARED ${ARROW_BUILD_SHARED}"
            "ARROW_GFLAGS_USE_SHARED ${ARROW_BUILD_SHARED}"
            "ARROW_GRPC_USE_SHARED ${ARROW_BUILD_SHARED}"
            "ARROW_PROTOBUF_USE_SHARED ${ARROW_BUILD_SHARED}"
            "ARROW_ZSTD_USE_SHARED ${ARROW_BUILD_SHARED}"
            "xsimd_SOURCE AUTO"
  )

  set(ARROW_FOUND TRUE)
  set(ARROW_LIBRARIES "")

  # Arrow_ADDED: set if CPM downloaded Arrow from Github Arrow_DIR:   set if CPM found Arrow on the
  # system/conda/etc.
  if(Arrow_ADDED OR Arrow_DIR)
    if(BUILD_STATIC)
      list(APPEND ARROW_LIBRARIES arrow_static)
      list(APPEND ARROW_LIBRARIES arrow_cuda_static)
    else()
      list(APPEND ARROW_LIBRARIES arrow_shared)
      list(APPEND ARROW_LIBRARIES arrow_cuda_shared)
    endif()

    if(Arrow_DIR)
      # Set this to enable `find_package(ArrowCUDA)`
      set(ArrowCUDA_DIR "${Arrow_DIR}")
      find_package(Arrow REQUIRED QUIET)
      find_package(ArrowCUDA REQUIRED QUIET)
      if(ENABLE_PARQUET)
        if(NOT Parquet_DIR)
          # Set this to enable `find_package(Parquet)`
          set(Parquet_DIR "${Arrow_DIR}")
        endif()
        # Set this to enable `find_package(ArrowDataset)`
        set(ArrowDataset_DIR "${Arrow_DIR}")
        find_package(ArrowDataset REQUIRED QUIET)
      endif()
    elseif(Arrow_ADDED)
      # Copy these files so we can avoid adding paths in Arrow_BINARY_DIR to
      # target_include_directories. That defeats ccache.
      file(INSTALL "${Arrow_BINARY_DIR}/src/arrow/util/config.h"
           DESTINATION "${Arrow_SOURCE_DIR}/cpp/src/arrow/util"
      )
      file(INSTALL "${Arrow_BINARY_DIR}/src/arrow/gpu/cuda_version.h"
           DESTINATION "${Arrow_SOURCE_DIR}/cpp/src/arrow/gpu"
      )
      if(ENABLE_PARQUET)
        file(INSTALL "${Arrow_BINARY_DIR}/src/parquet/parquet_version.h"
             DESTINATION "${Arrow_SOURCE_DIR}/cpp/src/parquet"
        )
      endif()
      #
      # This shouldn't be necessary!
      #
      # Arrow populates INTERFACE_INCLUDE_DIRECTORIES for the `arrow_static` and `arrow_shared`
      # targets in FindArrow and FindArrowCUDA respectively, so for static source-builds, we have to
      # do it after-the-fact.
      #
      # This only works because we know exactly which components we're using. Don't forget to update
      # this list if we add more!
      #
      foreach(ARROW_LIBRARY ${ARROW_LIBRARIES})
        target_include_directories(
          ${ARROW_LIBRARY}
          INTERFACE "$<BUILD_INTERFACE:${Arrow_SOURCE_DIR}/cpp/src>"
                    "$<BUILD_INTERFACE:${Arrow_SOURCE_DIR}/cpp/src/generated>"
                    "$<BUILD_INTERFACE:${Arrow_SOURCE_DIR}/cpp/thirdparty/hadoop/include>"
                    "$<BUILD_INTERFACE:${Arrow_SOURCE_DIR}/cpp/thirdparty/flatbuffers/include>"
        )
      endforeach()
    endif()
  else()
    set(ARROW_FOUND FALSE)
    message(FATAL_ERROR "CUDF: Arrow library not found or downloaded.")
  endif()

  if(Arrow_ADDED)

    set(arrow_code_string
        [=[
          if (TARGET cudf::arrow_shared AND (NOT TARGET arrow_shared))
              add_library(arrow_shared ALIAS cudf::arrow_shared)
          endif()
          if (TARGET arrow_shared AND (NOT TARGET cudf::arrow_shared))
              add_library(cudf::arrow_shared ALIAS arrow_shared)
          endif()
          if (TARGET cudf::arrow_static AND (NOT TARGET arrow_static))
              add_library(arrow_static ALIAS cudf::arrow_static)
          endif()
          if (TARGET arrow_static AND (NOT TARGET cudf::arrow_static))
              add_library(cudf::arrow_static ALIAS arrow_static)
          endif()
          if (NOT TARGET arrow::flatbuffers)
              add_library(arrow::flatbuffers INTERFACE IMPORTED)
          endif()
          if (NOT TARGET arrow::hadoop)
              add_library(arrow::hadoop INTERFACE IMPORTED)
          endif()
        ]=]
    )

    if(ENABLE_PARQUET)
      string(
        APPEND
        arrow_code_string
        [=[
          find_package(Boost)
          if (NOT TARGET Boost::headers)
              add_library(Boost::headers INTERFACE IMPORTED)
          endif()
        ]=]
      )
    endif()

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
      VERSION ${VERSION}
      EXPORT_SET arrow_targets
      GLOBAL_TARGETS arrow_shared arrow_static
      NAMESPACE cudf::
      FINAL_CODE_BLOCK arrow_code_string
    )

    set(arrow_cuda_code_string
        [=[
          if (TARGET cudf::arrow_cuda_shared AND (NOT TARGET arrow_cuda_shared))
              add_library(arrow_cuda_shared ALIAS cudf::arrow_cuda_shared)
          endif()
          if (TARGET arrow_cuda_shared AND (NOT TARGET cudf::arrow_cuda_shared))
              add_library(cudf::arrow_cuda_shared ALIAS arrow_cuda_shared)
          endif()
          if (TARGET cudf::arrow_cuda_static AND (NOT TARGET arrow_cuda_static))
              add_library(arrow_cuda_static ALIAS cudf::arrow_cuda_static)
          endif()
          if (TARGET arrow_cuda_static AND (NOT TARGET cudf::arrow_cuda_static))
              add_library(cudf::arrow_cuda_static ALIAS arrow_cuda_static)
          endif()
        ]=]
    )

    rapids_export(
      BUILD ArrowCUDA
      VERSION ${VERSION}
      EXPORT_SET arrow_cuda_targets
      GLOBAL_TARGETS arrow_cuda_shared arrow_cuda_static
      NAMESPACE cudf::
      FINAL_CODE_BLOCK arrow_cuda_code_string
    )

    if(ENABLE_PARQUET)

      set(arrow_dataset_code_string
          [=[
              if (TARGET cudf::arrow_dataset_shared AND (NOT TARGET arrow_dataset_shared))
                  add_library(arrow_dataset_shared ALIAS cudf::arrow_dataset_shared)
              endif()
              if (TARGET arrow_dataset_shared AND (NOT TARGET cudf::arrow_dataset_shared))
                  add_library(cudf::arrow_dataset_shared ALIAS arrow_dataset_shared)
              endif()
              if (TARGET cudf::arrow_dataset_static AND (NOT TARGET arrow_dataset_static))
                  add_library(arrow_dataset_static ALIAS cudf::arrow_dataset_static)
              endif()
              if (TARGET arrow_dataset_static AND (NOT TARGET cudf::arrow_dataset_static))
                  add_library(cudf::arrow_dataset_static ALIAS arrow_dataset_static)
              endif()
            ]=]
      )

      rapids_export(
        BUILD ArrowDataset
        VERSION ${VERSION}
        EXPORT_SET arrow_dataset_targets
        GLOBAL_TARGETS arrow_dataset_shared arrow_dataset_static
        NAMESPACE cudf::
        FINAL_CODE_BLOCK arrow_dataset_code_string
      )

      set(parquet_code_string
          [=[
              if (TARGET cudf::parquet_shared AND (NOT TARGET parquet_shared))
                  add_library(parquet_shared ALIAS cudf::parquet_shared)
              endif()
              if (TARGET parquet_shared AND (NOT TARGET cudf::parquet_shared))
                  add_library(cudf::parquet_shared ALIAS parquet_shared)
              endif()
              if (TARGET cudf::parquet_static AND (NOT TARGET parquet_static))
                  add_library(parquet_static ALIAS cudf::parquet_static)
              endif()
              if (TARGET parquet_static AND (NOT TARGET cudf::parquet_static))
                  add_library(cudf::parquet_static ALIAS parquet_static)
              endif()
            ]=]
      )

      rapids_export(
        BUILD Parquet
        VERSION ${VERSION}
        EXPORT_SET parquet_targets
        GLOBAL_TARGETS parquet_shared parquet_static
        NAMESPACE cudf::
        FINAL_CODE_BLOCK parquet_code_string
      )

      set(PROJECT_BINARY_DIR "${PROJECT_BINARY_DIR_prev}")
    endif()
  endif()
  # We generate the arrow-config and arrowcuda-config files when we built arrow locally, so always
  # do `find_dependency`
  rapids_export_package(BUILD Arrow cudf-exports)
  rapids_export_package(INSTALL Arrow cudf-exports)

  # We have to generate the find_dependency(ArrowCUDA) ourselves since we need to specify
  # ArrowCUDA_DIR to be where Arrow was found, since Arrow packages ArrowCUDA.config in a
  # non-standard location
  rapids_export_package(BUILD ArrowCUDA cudf-exports)
  if(ENABLE_PARQUET)
    rapids_export_package(BUILD Parquet cudf-exports)
    rapids_export_package(BUILD ArrowDataset cudf-exports)
  endif()

  include("${rapids-cmake-dir}/export/find_package_root.cmake")
  rapids_export_find_package_root(BUILD Arrow "${Arrow_BINARY_DIR}" cudf-exports)
  rapids_export_find_package_root(BUILD ArrowCUDA "${Arrow_BINARY_DIR}" cudf-exports)
  if(ENABLE_PARQUET)
    rapids_export_find_package_root(BUILD Parquet "${Arrow_BINARY_DIR}" cudf-exports)
    rapids_export_find_package_root(BUILD ArrowDataset "${Arrow_BINARY_DIR}" cudf-exports)
  endif()

  set(ARROW_FOUND
      "${ARROW_FOUND}"
      PARENT_SCOPE
  )
  set(ARROW_LIBRARIES
      "${ARROW_LIBRARIES}"
      PARENT_SCOPE
  )

  # Make sure consumers of our libs can see arrow libs
  _fix_cmake_global_defaults(arrow_shared)
  _fix_cmake_global_defaults(arrow_static)
  _fix_cmake_global_defaults(parquet_shared)
  _fix_cmake_global_defaults(parquet_static)
  _fix_cmake_global_defaults(arrow_cuda_shared)
  _fix_cmake_global_defaults(arrow_cuda_static)
  _fix_cmake_global_defaults(arrow_dataset_shared)
  _fix_cmake_global_defaults(arrow_dataset_static)

endfunction()

set(CUDF_VERSION_Arrow 8.0.0)

find_and_configure_arrow(
  ${CUDF_VERSION_Arrow}
  ON # ${CUDF_USE_ARROW_STATIC}
  OFF # ${CUDF_ENABLE_ARROW_S3}
  OFF # ${CUDF_ENABLE_ARROW_ORC}
  OFF # ${CUDF_ENABLE_ARROW_PYTHON}
  ON # ${CUDF_ENABLE_ARROW_PARQUET}
)
