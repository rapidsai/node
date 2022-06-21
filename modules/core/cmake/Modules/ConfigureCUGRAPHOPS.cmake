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

function(find_and_configure_cugraph_ops VERSION)

    include(get_cpm)

    include(ConfigureRAFT)

    _clean_build_dirs_if_not_fully_built(cugraph-ops libcugraph-ops++)

    _set_package_dir_if_exists(cugraph-ops cugraph-ops)

    if(NOT TARGET cugraph-ops::cugraph-ops++)
        if(DEFINED ENV{"CPM_cugraph-ops_SOURCE"})
            set("CPM_cugraph-ops_SOURCE" "$ENV{CPM_cugraph-ops_SOURCE}")
        endif()
        _get_major_minor_version(${VERSION} MAJOR_AND_MINOR)
        _get_update_disconnected_state(cugraph-ops ${VERSION} UPDATE_DISCONNECTED)
        CPMFindPackage(NAME     cugraph-ops
            VERSION             ${VERSION}
            GIT_REPOSITORY      "https://$ENV{RAPIDSAI_GITHUB_ACCESS_TOKEN}@github.com/trxcllnt/cugraph-ops.git"
            GIT_TAG             fea/enable-static-libs
            GIT_SHALLOW         TRUE
            ${UPDATE_DISCONNECTED}
            OPTIONS             "DETECT_CONDA_ENV OFF"
                                "BUILD_SHARED_LIBS OFF"
                                "BUILD_CUGRAPH_OPS_CPP_TESTS OFF")
    endif()
    # Make sure consumers of our libs can see cugraph-ops::cugraph-ops++
    _fix_cmake_global_defaults(cugraph-ops::cugraph-ops++)
    _fix_cmake_global_defaults(cugraph-ops::Thrust)
endfunction()

find_and_configure_cugraph_ops(${RMM_VERSION})
