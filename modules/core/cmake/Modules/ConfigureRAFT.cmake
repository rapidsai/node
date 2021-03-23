#=============================================================================
# Copyright (c) 2020, NVIDIA CORPORATION.
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

    CPMFindPackage(NAME raft
        VERSION        ${RAFT_VERSION}
        GIT_REPOSITORY https://github.com/rapidsai/raft.git
        GIT_TAG        ${RAFT_BRANCH}
        GIT_SHALLOW    TRUE
        DOWNLOAD_ONLY
    )

    set(RAFT_INCLUDE_DIR "${raft_SOURCE_DIR}/cpp/include" PARENT_SCOPE)
    message(STATUS "RAFT_INCLUDE_DIR: ${RAFT_INCLUDE_DIR}")
endfunction()

find_and_configure_raft(${RAFT_VERSION})
