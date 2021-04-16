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

function(find_and_configure_cugraph VERSION)

    include(get_cpm)

    _clean_build_dirs_if_not_fully_built(cugraph libcugraph.so)

    _set_package_dir_if_exists(cugraph cugraph)

    include(ConfigureCUDF)

    if(NOT TARGET cugraph::cugraph)

        include(ConfigureRAFT)

        # Have to set these in case configure and build steps are run separately
        # TODO: figure out why
        set(BUILD_TESTS OFF)
        set(BUILD_BENCHMARKS OFF)

        CPMFindPackage(NAME     cugraph
            VERSION             ${CUGRAPH_VERSION}
            GIT_REPOSITORY      https://github.com/rapidsai/cugraph.git
            GIT_TAG             branch-${CUGRAPH_VERSION}
            GIT_SHALLOW         TRUE
            UPDATE_DISCONNECTED FALSE
            # SOURCE_SUBDIR  cpp
            DOWNLOAD_ONLY
        )

        # set(CUGRAPH_INCLUDE_DIR_REAL "${cugraph_SOURCE_DIR}/cpp/include")
        # set(CUGRAPH_INCLUDE_DIR "${cugraph_SOURCE_DIR}/cpp/fake_include")
        # list(APPEND CUGRAPH_INCLUDE_DIRS ${RAFT_INCLUDE_DIR})
        # list(APPEND CUGRAPH_INCLUDE_DIRS ${CUGRAPH_INCLUDE_DIR})
        # list(APPEND CUGRAPH_INCLUDE_DIRS ${CUGRAPH_INCLUDE_DIR_REAL})
        # list(APPEND CUGRAPH_INCLUDE_DIRS ${cugraph_SOURCE_DIR}/cpp/src)
        # set(CUGRAPH_INCLUDE_DIRS ${CUGRAPH_INCLUDE_DIRS} PARENT_SCOPE)
        # set(cugraph_SOURCE_DIR "${cugraph_SOURCE_DIR}" PARENT_SCOPE)

        execute_process(COMMAND mkdir -p "${cugraph_SOURCE_DIR}/cpp/fake_include")
        execute_process(COMMAND ln -s -f "${cugraph_SOURCE_DIR}/cpp/include" "${cugraph_SOURCE_DIR}/cpp/fake_include/cugraph")

        # message(STATUS "CUGRAPH_INCLUDE_DIR: ${CUGRAPH_INCLUDE_DIR}")
        # message(STATUS "CUGRAPH_INCLUDE_DIR_REAL: ${CUGRAPH_INCLUDE_DIR_REAL}")
        # message(STATUS "CUGRAPH_INCLUDE_DIRS: ${CUGRAPH_INCLUDE_DIRS}")

        # synthesize a cugraph::cugraph target
        add_library(cugraph SHARED
            "${cugraph_SOURCE_DIR}/cpp/src/structure/graph.cu"
            "${cugraph_SOURCE_DIR}/cpp/src/layout/force_atlas2.cu")

        set_target_properties(cugraph
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

        target_include_directories(cugraph
            PUBLIC "${cugraph_SOURCE_DIR}/cpp/fake_include"
                   "${cugraph_SOURCE_DIR}/cpp/include"
                   "${cugraph_SOURCE_DIR}/cpp/src")

        set(NODE_RAPIDS_CUGRAPH_CUDA_FLAGS ${NODE_RAPIDS_CMAKE_CUDA_FLAGS})
        list(APPEND NODE_RAPIDS_CUGRAPH_CUDA_FLAGS -Xptxas --disable-warnings)
        list(APPEND NODE_RAPIDS_CUGRAPH_CUDA_FLAGS -Xcompiler=-Wall,-Wno-error=sign-compare,-Wno-error=unused-but-set-variable)

        target_compile_options(cugraph
            PRIVATE "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CUDA>:${NODE_RAPIDS_CUGRAPH_CUDA_FLAGS}>>"
        )

        target_link_libraries(cugraph PUBLIC raft::raft CUDA::cudart_static)

        add_library(cugraph::cugraph ALIAS cugraph)
    endif()
endfunction()

find_and_configure_cugraph(${CUGRAPH_VERSION})
