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

    include(ConfigureRAFT)

    _clean_build_dirs_if_not_fully_built(cugraph libcugraph.so)

    _set_package_dir_if_exists(cuco cuco)
    _set_package_dir_if_exists(faiss faiss)
    _set_package_dir_if_exists(cugraph cugraph)
    _set_package_dir_if_exists(cuhornet cuhornet)

    if(NOT TARGET cugraph::cugraph)
        _fix_rapids_cmake_dir()
        _get_major_minor_version(${VERSION} MAJOR_AND_MINOR)
        _get_update_disconnected_state(cugraph ${VERSION} UPDATE_DISCONNECTED)
        CPMFindPackage(NAME     cugraph
            VERSION             ${VERSION}
            GIT_REPOSITORY      https://github.com/rapidsai/cugraph.git
            GIT_TAG             branch-${MAJOR_AND_MINOR}
            GIT_SHALLOW         TRUE
            ${UPDATE_DISCONNECTED}
            SOURCE_SUBDIR       cpp
            OPTIONS             "BUILD_TESTS OFF"
                                "BUILD_BENCHMARKS OFF"
        )
        _fix_rapids_cmake_dir()
    endif()
    # Make sure consumers of our libs can see cugraph::cugraph
    _fix_cmake_global_defaults(cugraph::cugraph)
endfunction()

find_and_configure_cugraph(${CUGRAPH_VERSION})
