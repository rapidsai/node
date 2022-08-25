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

function(find_and_configure_raft)

    include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/get_cpm.cmake)
    include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/get_version.cmake)
    include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/ConfigureRMM.cmake)

    _get_rapidsai_module_version(raft VERSION)

    _clean_build_dirs_if_not_fully_built(raft libraft_nn)
    _clean_build_dirs_if_not_fully_built(raft libraft_distance)

    _set_thrust_dir_if_exists()
    _set_package_dir_if_exists(rmm rmm)
    _set_package_dir_if_exists(raft raft)
    _set_package_dir_if_exists(faiss faiss)

    if(NOT TARGET raft::raft)
        _get_major_minor_version(${VERSION} MAJOR_AND_MINOR)
        _get_update_disconnected_state(raft ${VERSION} UPDATE_DISCONNECTED)
        CPMFindPackage(NAME        raft
            VERSION                ${VERSION}
            GIT_REPOSITORY         https://github.com/rapidsai/raft.git
            GIT_TAG                branch-${MAJOR_AND_MINOR}
            GIT_SHALLOW            TRUE
            SOURCE_SUBDIR          cpp
            FIND_PACKAGE_ARGUMENTS "COMPONENTS distance nn"
            ${UPDATE_DISCONNECTED}
            OPTIONS                "BUILD_TESTS OFF"
                                   "BUILD_SHARED_LIBS OFF"
                                   "CUDA_STATIC_RUNTIME ON"
                                   "RAFT_USE_FAISS_STATIC ON"
                                   "RAFT_COMPILE_LIBRARIES ON")
    endif()
    # Make sure consumers of our libs can see raft::raft
    _fix_cmake_global_defaults(raft::raft)
    # Make these -isystem so -Werror doesn't fail their builds
    _set_interface_include_dirs_as_system(faiss::faiss)

    get_target_property(_link_libs faiss::faiss INTERFACE_LINK_LIBRARIES)
    string(REPLACE "CUDA::cudart" "CUDA::cudart_static" _link_libs "${_link_libs}")
    string(REPLACE "CUDA::cublas" "CUDA::cublas_static" _link_libs "${_link_libs}")
    set_target_properties(faiss::faiss PROPERTIES INTERFACE_LINK_LIBRARIES "${_link_libs}")

    get_target_property(_link_libs raft::raft INTERFACE_LINK_LIBRARIES)
    string(REPLACE "CUDA::cufft" "CUDA::cufft_static" _link_libs "${_link_libs}")
    string(REPLACE "CUDA::cublas" "CUDA::cublas_static" _link_libs "${_link_libs}")
    string(REPLACE "CUDA::curand" "CUDA::curand_static" _link_libs "${_link_libs}")
    string(REPLACE "CUDA::cusolver" "CUDA::cusolver_static" _link_libs "${_link_libs}")
    string(REPLACE "CUDA::cudart" "CUDA::cudart_static" _link_libs "${_link_libs}")
    string(REPLACE "CUDA::cusparse" "CUDA::cusparse_static" _link_libs "${_link_libs}")
    set_target_properties(raft::raft PROPERTIES INTERFACE_LINK_LIBRARIES "${_link_libs}")

    set(raft_VERSION "${raft_VERSION}" PARENT_SCOPE)
endfunction()

find_and_configure_raft()
