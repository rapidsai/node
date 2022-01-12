#=============================================================================
# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

function(find_and_configure_cudf VERSION)

    include(get_cpm)

    include(ConfigureRMM)

    _clean_build_dirs_if_not_fully_built(cudf libcudf.so)

    _set_package_dir_if_exists(cudf cudf)
    _set_package_dir_if_exists(dlpack dlpack)
    _set_package_dir_if_exists(jitify jitify)
    _set_package_dir_if_exists(nvcomp nvcomp)
    _set_package_dir_if_exists(Thrust thrust)

    # Set this so Arrow doesn't add `-Werror` to
    # CMAKE_CXX_FLAGS when CMAKE_BUILD_TYPE=Debug
    set(BUILD_WARNING_LEVEL "PRODUCTION")
    set(BUILD_WARNING_LEVEL "PRODUCTION" PARENT_SCOPE)
    set(BUILD_WARNING_LEVEL "PRODUCTION" CACHE STRING "" FORCE)

    if(NOT TARGET cudf::cudf)
        _get_major_minor_version(${VERSION} MAJOR_AND_MINOR)
        _get_update_disconnected_state(cudf ${VERSION} UPDATE_DISCONNECTED)
        CPMFindPackage(NAME     cudf
            VERSION             ${VERSION}
            # GIT_REPOSITORY      https://github.com/rapidsai/cudf.git
            # GIT_TAG             branch-${MAJOR_AND_MINOR}
            GIT_REPOSITORY      https://github.com/trxcllnt/cudf.git
            GIT_TAG             fix/nvcomp-absolute-paths
            GIT_SHALLOW         TRUE
            ${UPDATE_DISCONNECTED}
            SOURCE_SUBDIR       cpp
            OPTIONS             "BUILD_TESTS OFF"
                                "BUILD_BENCHMARKS OFF"
                                "JITIFY_USE_CACHE ON"
                                "BOOST_SOURCE SYSTEM"
                                "Thrift_SOURCE BUNDLED"
                                "CUDA_STATIC_RUNTIME OFF"
                                "CUDF_USE_ARROW_STATIC OFF"
                                "CUDF_ENABLE_ARROW_S3 OFF"
                                "CUDF_ENABLE_ARROW_ORC OFF"
                                "CUDF_ENABLE_ARROW_PYTHON OFF"
                                "CUDF_ENABLE_ARROW_PARQUET ON"
                                "PER_THREAD_DEFAULT_STREAM ON"
                                "DISABLE_DEPRECATION_WARNING ON")
    endif()

    # Make sure consumers of our libs can see cudf::cudf
    _fix_cmake_global_defaults(cudf::cudf)
    # Make sure consumers of our libs can see cudf::cudftestutil
    _fix_cmake_global_defaults(cudf::cudftestutil)
endfunction()

find_and_configure_cudf(${CUDF_VERSION})
