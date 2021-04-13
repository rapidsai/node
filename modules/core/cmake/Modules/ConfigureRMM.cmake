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

_set_package_dir_if_exists(rmm rmm)
_set_package_dir_if_exists(spdlog spdlog)
_set_package_dir_if_exists(Thrust thrust)

function(find_and_configure_rmm VERSION)

    if(NOT TARGET rmm::rmm)

        if (NOT DEFINED ENV{NODE_RAPIDS_USE_LOCAL_DEPS_BUILD_DIRS})
            # if (EXISTS "${FETCHCONTENT_BASE_DIR}/spdlog-build")
            #     file(REMOVE_RECURSE "${FETCHCONTENT_BASE_DIR}/spdlog-build")
            #     file(REMOVE_RECURSE "${FETCHCONTENT_BASE_DIR}/spdlog-subbuild")
            # endif()
            if (EXISTS "${FETCHCONTENT_BASE_DIR}/thrust-subbuild")
                file(REMOVE_RECURSE "${FETCHCONTENT_BASE_DIR}/thrust-subbuild")
            endif()
        endif()

        CPMFindPackage(NAME     rmm
            VERSION             ${RMM_VERSION}
            GIT_REPOSITORY      https://github.com/rapidsai/rmm.git
            GIT_TAG             branch-${RMM_VERSION}
            GIT_SHALLOW         TRUE
            UPDATE_DISCONNECTED FALSE
            OPTIONS             "BUILD_TESTS OFF"
                                "BUILD_BENCHMARKS OFF"
                                "CUDA_STATIC_RUNTIME ON"
                                "DISABLE_DEPRECATION_WARNING ${DISABLE_DEPRECATION_WARNINGS}")
    endif()

    # Make sure consumers of our libs can see rmm::rmm
    _fix_cmake_global_defaults(rmm::rmm)
    _fix_cmake_global_defaults(rmm::Thrust)
    _fix_cmake_global_defaults(rmm::spdlog_header_only)
endfunction()

find_and_configure_rmm(${RMM_VERSION})
