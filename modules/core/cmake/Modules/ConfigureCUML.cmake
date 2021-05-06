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

        set(SINGLEGPU ON)
        set(WITH_UCX OFF)
        set(DISABLE_OPENMP OFF)
        set(DETECT_CONDA_ENV OFF)
        set(ENABLE_CUMLPRIMS_MG OFF)
        set(BUILD_CUML_MG_TESTS OFF)
        set(BUILD_CUML_STD_COMMS OFF)
        set(BUILD_CUML_MPI_COMMS OFF)

        set(BUILD_CUML_TESTS OFF)
        set(BUILD_CUML_BENCH OFF)
        set(BUILD_PRIMS_TESTS OFF)
        set(BUILD_CUML_EXAMPLES OFF)
        set(BUILD_CUML_C_LIBRARY OFF)
        set(BUILD_CUML_CPP_LIBRARY OFF)

        CPMFindPackage(NAME     cuml
            VERSION             ${CUML_VERSION}
            GIT_REPOSITORY      https://github.com/rapidsai/cuml.git
            GIT_TAG             branch-${CUML_VERSION}
            GIT_SHALLOW         TRUE
            UPDATE_DISCONNECTED FALSE
            # SOURCE_SUBDIR  cpp
            DOWNLOAD_ONLY
        )

        # synthesize a cuml::cuml target
        add_library(cuml SHARED
            "${cuml_SOURCE_DIR}/cpp/src/umap/umap.cu")

        set_target_properties(cuml
            PROPERTIES BUILD_RPATH                         "\$ORIGIN"
                       INSTALL_RPATH                       "\$ORIGIN"
                       # set target compile options
                       CXX_STANDARD                        17
                       CXX_STANDARD_REQUIRED               ON
                       CUDA_STANDARD                       17
                       CUDA_STANDARD_REQUIRED              ON
                       NO_SYSTEM_FROM_IMPORTED             ON
                       POSITION_INDEPENDENT_CODE           ON
                       INTERFACE_POSITION_INDEPENDENT_CODE ON
        )

        target_include_directories(cuml
            PUBLIC "${cuml_SOURCE_DIR}/cpp/include"
                   "${cuml_SOURCE_DIR}/cpp/src_prims"
                   "${cuml_SOURCE_DIR}/cpp/src")

        set(NODE_RAPIDS_CUML_CUDA_FLAGS ${NODE_RAPIDS_CMAKE_CUDA_FLAGS})

        list(APPEND NODE_RAPIDS_CUML_CUDA_FLAGS -Xcudafe --diag_suppress=unrecognized_gcc_pragma -Xfatbin=-compress-all)

        target_compile_options(cuml
            PRIVATE "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CUDA>:${NODE_RAPIDS_CUML_CUDA_FLAGS}>>"
        )

        target_link_libraries(cuml PUBLIC raft::raft CUDA::cudart_static)

        add_library(cuml::cuml ALIAS cuml)
    endif()
endfunction()

find_and_configure_cuml(${CUML_VERSION})
