#=============================================================================
# Copyright (c) 2021, NVIDIA CORPORATION.
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

function(find_and_configure_cuml VERSION)

    include(get_cpm)

    _clean_build_dirs_if_not_fully_built(cuml libcuml.so)

    _set_package_dir_if_exists(cuml cuml)

    include(ConfigureCUDF)

    if(NOT TARGET cuml::cuml)

        include(ConfigureRAFT)

        # Have to set these in case configure and build steps are run separately
        # TODO: figure out why

        # set(SINGLEGPU ON)
        # set(WITH_UCX OFF)
        # set(DISABLE_OPENMP OFF)
        # set(DETECT_CONDA_ENV OFF)
        # set(ENABLE_CUMLPRIMS_MG OFF)
        # set(BUILD_CUML_MG_TESTS OFF)
        # set(BUILD_CUML_STD_COMMS OFF)
        # set(BUILD_CUML_MPI_COMMS OFF)

        # set(BUILD_CUML_TESTS OFF)
        # set(BUILD_CUML_BENCH OFF)
        # set(BUILD_PRIMS_TESTS OFF)
        # set(BUILD_CUML_EXAMPLES OFF)
        # set(BUILD_CUML_C_LIBRARY OFF)
        # set(BUILD_CUML_CPP_LIBRARY OFF)

        CPMFindPackage(NAME     cuml
            VERSION             ${CUML_VERSION}
            # GIT_REPOSITORY      https://github.com/rapidsai/cuml.git
            # GIT_TAG             branch-${CUML_VERSION}
            GIT_REPOSITORY      https://github.com/dantegd/cuml.git
            GIT_TAG             fea-rapids-cmake
            GIT_SHALLOW         TRUE
            UPDATE_DISCONNECTED FALSE
            SOURCE_SUBDIR       cpp
            OPTIONS             "SINGLEGPU ON"
                                "WITH_UCX OFF"
                                "DISABLE_OPENMP OFF"
                                "DETECT_CONDA_ENV OFF"
                                "ENABLE_CUMLPRIMS_MG OFF"
                                "BUILD_CUML_MG_TESTS OFF"
                                "BUILD_CUML_STD_COMMS OFF"
                                "BUILD_CUML_MPI_COMMS OFF"
                                "BUILD_CUML_TESTS OFF"
                                "BUILD_CUML_BENCH OFF"
                                "BUILD_PRIMS_TESTS OFF"
                                "BUILD_CUML_EXAMPLES OFF"
                                "BUILD_CUML_C_LIBRARY OFF"
                                "BUILD_CUML_CPP_LIBRARY ON"
        )

        # Make sure consumers of our libs can see cuml::cuml++
        _fix_cmake_global_defaults(cuml::cuml++)
    endif()
endfunction()

find_and_configure_cuml(${CUML_VERSION})
