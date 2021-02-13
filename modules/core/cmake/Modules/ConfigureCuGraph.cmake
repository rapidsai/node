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

function(find_and_configure_cugraph VERSION)

    include(get_cpm)
    include(ConfigureRAFT)

    CPMAddPackage(NAME cugraph
        VERSION        ${CUGRAPH_VERSION}
        GIT_REPOSITORY https://github.com/rapidsai/cugraph.git
        GIT_TAG        branch-${CUGRAPH_VERSION}
        GIT_SHALLOW    TRUE
        DOWNLOAD_ONLY
    )

    set(CUGRAPH_INCLUDE_DIR_REAL "${cugraph_SOURCE_DIR}/cpp/include")
    set(CUGRAPH_INCLUDE_DIR "${cugraph_SOURCE_DIR}/cpp/fake_include")

    list(APPEND CUGRAPH_INCLUDE_DIRS ${RAFT_INCLUDE_DIR})
    list(APPEND CUGRAPH_INCLUDE_DIRS ${CUGRAPH_INCLUDE_DIR})
    list(APPEND CUGRAPH_INCLUDE_DIRS ${CUGRAPH_INCLUDE_DIR_REAL})
    list(APPEND CUGRAPH_INCLUDE_DIRS ${cugraph_SOURCE_DIR}/cpp/src)
    set(CUGRAPH_INCLUDE_DIRS ${CUGRAPH_INCLUDE_DIRS} PARENT_SCOPE)
    set(cugraph_SOURCE_DIR ${cugraph_SOURCE_DIR} PARENT_SCOPE)

    execute_process(COMMAND mkdir -p ${CUGRAPH_INCLUDE_DIR})
    execute_process(COMMAND ln -s -f ${CUGRAPH_INCLUDE_DIR_REAL} ${CUGRAPH_INCLUDE_DIR}/cugraph)

    message(STATUS "CUGRAPH_INCLUDE_DIR: ${CUGRAPH_INCLUDE_DIR}")
    message(STATUS "CUGRAPH_INCLUDE_DIR_REAL: ${CUGRAPH_INCLUDE_DIR_REAL}")
    message(STATUS "CUGRAPH_INCLUDE_DIRS: ${CUGRAPH_INCLUDE_DIRS}")
endfunction()

find_and_configure_cugraph(${CUGRAPH_VERSION})
