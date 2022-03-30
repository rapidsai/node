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

function(find_and_configure_raft VERSION)

    include(get_cpm)

    include(ConfigureRMM)

    _clean_build_dirs_if_not_fully_built(raft libraft_nn)
    _clean_build_dirs_if_not_fully_built(raft libraft_distance)

    _set_package_dir_if_exists(rmm rmm)
    _set_package_dir_if_exists(raft raft)
    _set_package_dir_if_exists(faiss faiss)
    _set_package_dir_if_exists(Thrust thrust)

    if(NOT TARGET raft::raft)
        _get_major_minor_version(${VERSION} MAJOR_AND_MINOR)
        _get_update_disconnected_state(raft ${VERSION} UPDATE_DISCONNECTED)
        CPMFindPackage(NAME        raft
            VERSION                ${VERSION}
            GIT_REPOSITORY         https://github.com/trxcllnt/raft.git
            GIT_TAG                fea/enable-static-libs
            GIT_SHALLOW            TRUE
            SOURCE_SUBDIR          cpp
            FIND_PACKAGE_ARGUMENTS "COMPONENTS distance nn"
            ${UPDATE_DISCONNECTED}
            OPTIONS                "BUILD_TESTS OFF"
                                   "BUILD_SHARED_LIBS OFF"
                                   "RAFT_USE_FAISS_STATIC ON")
    endif()
    # if (TARGET raft_nn_lib)
    #     set_target_properties(raft_nn_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
    #     target_compile_options(raft_nn_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-fPIC>)
    # endif()
    # if (TARGET raft_distance_lib)
    #     set_target_properties(raft_distance_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
    #     target_compile_options(raft_distance_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-fPIC>)
    # endif()
    # Make sure consumers of our libs can see raft::raft
    _fix_cmake_global_defaults(raft::raft)
endfunction()

find_and_configure_raft(${RMM_VERSION})
