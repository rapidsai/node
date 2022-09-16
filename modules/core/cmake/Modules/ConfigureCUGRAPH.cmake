#=============================================================================
# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

function(find_and_configure_cugraph)

    include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/get_cpm.cmake)
    include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/get_nccl.cmake)
    include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/get_version.cmake)
    include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/ConfigureCUGRAPHOPS.cmake)

    _get_rapidsai_module_version(cugraph VERSION)

    _clean_build_dirs_if_not_fully_built(cugraph libcugraph)

    _set_thrust_dir_if_exists()
    _set_package_dir_if_exists(cuco cuco)
    _set_package_dir_if_exists(raft raft)
    _set_package_dir_if_exists(cugraph cugraph)
    _set_package_dir_if_exists(cuhornet cuhornet)
    _set_package_dir_if_exists(cugraph-ops cugraph-ops)

    if(NOT TARGET cugraph::cugraph)
        _get_major_minor_version(${VERSION} MAJOR_AND_MINOR)
        _get_update_disconnected_state(cugraph ${VERSION} UPDATE_DISCONNECTED)
        CPMFindPackage(NAME        cugraph
            VERSION                ${VERSION}
            GIT_REPOSITORY         https://github.com/rapidsai/cugraph.git
            GIT_TAG                branch-${MAJOR_AND_MINOR}
            # EXCLUDE_FROM_ALL       TRUE
            GIT_SHALLOW            TRUE
            ${UPDATE_DISCONNECTED}
            SOURCE_SUBDIR          cpp
            OPTIONS                "BUILD_TESTS OFF"
                                   "BUILD_BENCHMARKS OFF"
                                   "BUILD_SHARED_LIBS OFF"
                                   "CUDA_STATIC_RUNTIME ON"
                                   "BUILD_CUGRAPH_MG_TESTS OFF"
        )
    endif()
    # Make sure consumers of our libs can see cugraph::cugraph
    _fix_cmake_global_defaults(cugraph::cugraph)

    if(NOT TARGET cugraph::cuHornet AND
      (NOT DEFINED ENV{NODE_RAPIDS_USE_LOCAL_DEPS_BUILD_DIRS}))
        set(cuhornet_SOURCE_DIR "${CPM_BINARY_CACHE}/cuhornet-src")
        if (EXISTS "${cuhornet_SOURCE_DIR}")
            add_library(cugraph::cuHornet IMPORTED INTERFACE GLOBAL)
            target_include_directories(cugraph::cuHornet INTERFACE
                "${cuhornet_SOURCE_DIR}/hornet/include"
                "${cuhornet_SOURCE_DIR}/hornetsnest/include"
                "${cuhornet_SOURCE_DIR}/xlib/include"
                "${cuhornet_SOURCE_DIR}/primitives"
            )
        endif()
    endif()

    include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/link_utils.cmake)
    _statically_link_cuda_toolkit_libs(cugraph::cugraph)

    set(cugraph_VERSION "${cugraph_VERSION}" PARENT_SCOPE)
endfunction()

find_and_configure_cugraph()
