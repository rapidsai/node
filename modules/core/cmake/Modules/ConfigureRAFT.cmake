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

function(find_and_configure_raft VERSION)

    include(get_cpm)

    _set_package_dir_if_exists(rmm rmm)
    _set_package_dir_if_exists(ucx ucx)
    _set_package_dir_if_exists(raft raft)
    _set_package_dir_if_exists(cuco cuco)
    _set_package_dir_if_exists(nccl nccl)
    _set_package_dir_if_exists(faiss faiss)
    _set_package_dir_if_exists(Thrust thrust)
    _set_package_dir_if_exists(libcudacxx libcudacxx)

    _fix_rapids_cmake_dir()

    if(NOT TARGET raft::raft)

        # We only want to set `UPDATE_DISCONNECTED` while
        # the GIT tag hasn't moved from the last time we cloned
        set(cpm_raft_disconnect_update "UPDATE_DISCONNECTED TRUE")
        set(CPM_RAFT_CURRENT_VERSION ${VERSION} CACHE STRING "version of raft we checked out")
        if(NOT VERSION VERSION_EQUAL CPM_RAFT_CURRENT_VERSION)
            set(CPM_RAFT_CURRENT_VERSION ${VERSION} CACHE STRING "version of raft we checked out" FORCE)
            set(cpm_raft_disconnect_update "")
        endif()

        if(${VERSION} MATCHES [=[([0-9]+)\.([0-9]+)\.([0-9]+)]=])
            set(MAJOR_AND_MINOR "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")
        else()
            set(MAJOR_AND_MINOR "${VERSION}")
        endif()

        CPMAddPackage(NAME raft
            VERSION        ${VERSION}
            # GIT_REPOSITORY https://github.com/rapidsai/raft.git
            # GIT_TAG        branch-${MAJOR_AND_MINOR}
            GIT_REPOSITORY https://github.com/cjnolet/raft.git
            GIT_TAG        bug-sparse_bfknn_const_handle
            GIT_SHALLOW    TRUE
            SOURCE_SUBDIR  cpp
            ${cpm_raft_disconnect_update}
            OPTIONS        "BUILD_TESTS OFF"
        )

        # Make sure consumers of our libs can see raft::raft
        _fix_cmake_global_defaults(raft::raft)
    endif()

    _fix_rapids_cmake_dir()

endfunction()

find_and_configure_raft(${RAFT_VERSION})
