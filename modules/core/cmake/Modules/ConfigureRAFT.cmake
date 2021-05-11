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

function(find_and_configure_raft VERSION)

    include(get_cpm)

    _set_package_dir_if_exists(raft raft)

    if(NOT TARGET raft::raft)

        include(ConfigureRMM)

        # Have to set these in case configure and build steps are run separately
        # TODO: figure out why
        set(BUILD_RAFT_TESTS OFF)

        CPMFindPackage(NAME     raft
            VERSION             ${VERSION}
            GIT_REPOSITORY      ${RAFT_GITHUB_URL}
            GIT_TAG             ${RAFT_BRANCH}
            DOWNLOAD_ONLY
        )

        # Synthesize a raft::raft target
        add_library(raft INTERFACE)
        target_include_directories(raft
            INTERFACE "$<BUILD_INTERFACE:${raft_SOURCE_DIR}/cpp/include>")

        target_link_libraries(raft
            INTERFACE rmm::rmm
                      CUDA::cublas
                      CUDA::cusolver
                      CUDA::cusparse
                      CUDA::curand
                      CUDA::cudart_static)

        add_library(raft::raft ALIAS raft)

        install(TARGETS raft
                DESTINATION lib
                EXPORT raft-exports)

        install(EXPORT raft-exports
                DESTINATION "lib/cmake/raft"
                FILE raft-targets.cmake
                NAMESPACE raft::)

        export(EXPORT raft-exports
               FILE raft-targets.cmake
               NAMESPACE raft::)

        # Make sure consumers of our libs can see raft::raft
        _fix_cmake_global_defaults(raft::raft)
    endif()
endfunction()

find_and_configure_raft(${RAFT_VERSION})
