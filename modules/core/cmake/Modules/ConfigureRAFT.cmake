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

include(get_cpm)

_set_package_dir_if_exists(raft raft)

function(find_and_configure_raft VERSION)

    if(NOT TARGET raft::raft)

        include(ConfigureRMM)

        # Have to set these in case configure and build steps are run separately
        # TODO: figure out why
        set(BUILD_RAFT_TESTS OFF)

        CPMFindPackage(NAME raft
            VERSION        ${VERSION}
            GIT_REPOSITORY https://github.com/rapidsai/raft.git
            GIT_TAG        ${RAFT_BRANCH}
            GIT_SHALLOW    TRUE
            # SOURCE_SUBDIR  cpp
            DOWNLOAD_ONLY
        )

        # Synthesize a raft::raft target
        add_library(raft INTERFACE)
        target_include_directories(raft INTERFACE "${raft_SOURCE_DIR}/cpp/include")
        target_link_libraries(raft
            INTERFACE rmm::rmm
                      CUDA::cublas
                      CUDA::cusolver
                      CUDA::cusparse
                      CUDA::curand
                      CUDA::cudart_static)

        add_library(raft::raft ALIAS raft)
    endif()
endfunction()

find_and_configure_raft(${RAFT_VERSION})
