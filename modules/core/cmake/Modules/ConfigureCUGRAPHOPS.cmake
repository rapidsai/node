#=============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
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

function(find_and_configure_cugraph_ops)

    include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/get_cpm.cmake)
    include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/get_version.cmake)
    include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/ConfigureRAFT.cmake)

    _get_rapidsai_module_version(cugraph-ops VERSION)

    _clean_build_dirs_if_not_fully_built(cugraph-ops libcugraph-ops++)

    _set_thrust_dir_if_exists()
    _set_package_dir_if_exists(raft raft)
    _set_package_dir_if_exists(cugraph-ops cugraph-ops)

    if(NOT TARGET cugraph-ops::cugraph-ops++)
        _get_major_minor_version(${VERSION} MAJOR_AND_MINOR)
        _get_update_disconnected_state(cugraph-ops ${VERSION} UPDATE_DISCONNECTED)
        CPMFindPackage(NAME     cugraph-ops
            VERSION             ${VERSION}
            GIT_REPOSITORY      "git@github.com:rapidsai/cugraph-ops.git"
            GIT_TAG             branch-${MAJOR_AND_MINOR}
            GIT_SHALLOW         TRUE
            ${UPDATE_DISCONNECTED}
            SOURCE_SUBDIR       cpp
            OPTIONS             "DETECT_CONDA_ENV OFF"
                                "BUILD_SHARED_LIBS OFF"
                                "CUDA_STATIC_RUNTIME ON"
                                "BUILD_CUGRAPH_OPS_CPP_TESTS OFF")
    endif()
    # Make sure consumers of our libs can see cugraph-ops::Thrust
    _fix_cmake_global_defaults(cugraph-ops::Thrust)
    # Make sure consumers of our libs can see cugraph-ops::cugraph-ops++
    _fix_cmake_global_defaults(cugraph-ops::cugraph-ops++)

    get_target_property(_link_libs cugraph-ops::cugraph-ops++ INTERFACE_LINK_LIBRARIES)
    string(REPLACE "CUDA::cudart" "CUDA::cudart_static" _link_libs "${_link_libs}")
    string(REPLACE "CUDA::cublas" "CUDA::cublas_static" _link_libs "${_link_libs}")
    string(REPLACE "CUDA::curand" "CUDA::curand_static" _link_libs "${_link_libs}")
    set_target_properties(cugraph-ops::cugraph-ops++ PROPERTIES INTERFACE_LINK_LIBRARIES "${_link_libs}")

    set(cugraph-ops_VERSION "${cugraph-ops_VERSION}" PARENT_SCOPE)
endfunction()

find_and_configure_cugraph_ops()
