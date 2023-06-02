#=============================================================================
# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

function(find_and_configure_cudf)

    include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/get_cpm.cmake)
    include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/get_version.cmake)
    include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/ConfigureRMM.cmake)
    include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/ConfigureArrow.cmake)

    _get_rapidsai_module_version(cudf VERSION)

    _clean_build_dirs_if_not_fully_built(cudf libcudf)
    _clean_build_dirs_if_not_fully_built(nvcomp libnvcomp)

    _set_thrust_dir_if_exists()
    _set_package_dir_if_exists(cudf cudf)
    _set_package_dir_if_exists(cuco cuco)
    _set_package_dir_if_exists(dlpack dlpack)
    _set_package_dir_if_exists(jitify jitify)
    _set_package_dir_if_exists(nvcomp nvcomp)
    _set_package_dir_if_exists(Arrow arrow)
    _set_package_dir_if_exists(Parquet arrow)
    _set_package_dir_if_exists(ArrowCUDA arrow)
    _set_package_dir_if_exists(ArrowDataset arrow)

    if(NOT TARGET cudf::cudf)
        _get_major_minor_version(${VERSION} MAJOR_AND_MINOR)
        _get_update_disconnected_state(cudf ${VERSION} UPDATE_DISCONNECTED)
        CPMFindPackage(NAME     cudf
            VERSION             ${VERSION}
            # EXCLUDE_FROM_ALL    TRUE
            GIT_REPOSITORY      https://github.com/rapidsai/cudf.git
            GIT_TAG             branch-${MAJOR_AND_MINOR}
            GIT_SHALLOW         TRUE
            ${UPDATE_DISCONNECTED}
            SOURCE_SUBDIR       cpp
            OPTIONS             "BUILD_TESTS OFF"
                                "BUILD_BENCHMARKS OFF"
                                "BUILD_SHARED_LIBS OFF"
                                "JITIFY_USE_CACHE ON"
                                "BOOST_SOURCE SYSTEM"
                                "Thrift_SOURCE BUNDLED"
                                "CUDA_STATIC_RUNTIME ON"
                                "CUDF_USE_ARROW_STATIC ON"
                                "CUDF_ENABLE_ARROW_S3 OFF"
                                # "CUDF_ENABLE_ARROW_S3 ON"
                                "CUDF_ENABLE_ARROW_ORC OFF"
                                "CUDF_ENABLE_ARROW_PYTHON OFF"
                                "CUDF_ENABLE_ARROW_PARQUET ON"
                                # "ARROW_DEPENDENCY_SOURCE AUTO"
                                "DISABLE_DEPRECATION_WARNING ON"
                                "CUDF_USE_PROPRIETARY_NVCOMP OFF"
                                "CUDF_USE_PER_THREAD_DEFAULT_STREAM ON")
    endif()

    set(cudf_VERSION "${cudf_VERSION}" PARENT_SCOPE)
    set(ARROW_LIBRARIES ${ARROW_LIBRARIES} PARENT_SCOPE)

    _set_package_dir_if_exists(nvcomp nvcomp)
    find_package(nvcomp)
    set_target_properties(nvcomp
        PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${nvcomp_ROOT}"
                   LIBRARY_OUTPUT_DIRECTORY "${nvcomp_ROOT}")

    include(CMakePackageConfigHelpers)
    write_basic_package_version_file(
      ${nvcomp_ROOT}/nvcomp-config-version.cmake
      VERSION 2.3
      COMPATIBILITY ExactVersion)

    include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/link_utils.cmake)
    _statically_link_cuda_toolkit_libs(cudf::cudf)
    _statically_link_cuda_toolkit_libs(cudf::cudftestutil)

endfunction()

find_and_configure_cudf()
