#=============================================================================
# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

function(find_and_configure_rmm)

    include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/get_cpm.cmake)
    include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/get_version.cmake)
    include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/ConfigureThrust.cmake)

    _get_rapidsai_module_version(rmm VERSION)

    _set_thrust_dir_if_exists()
    _set_package_dir_if_exists(rmm rmm)
    _set_package_dir_if_exists(spdlog spdlog)

    if(NOT TARGET rmm::rmm)
        _get_major_minor_version(${VERSION} MAJOR_AND_MINOR)
        _get_update_disconnected_state(rmm ${VERSION} UPDATE_DISCONNECTED)
        CPMFindPackage(NAME     rmm
            VERSION             ${VERSION}
            GIT_REPOSITORY      https://github.com/rapidsai/rmm.git
            GIT_TAG             branch-${MAJOR_AND_MINOR}
            GIT_SHALLOW         TRUE
            ${UPDATE_DISCONNECTED}
            OPTIONS             "BUILD_TESTS OFF"
                                "BUILD_BENCHMARKS OFF"
                                "DISABLE_DEPRECATION_WARNING ${DISABLE_DEPRECATION_WARNINGS}")
    endif()
    # Make sure consumers of our libs can see rmm::rmm
    _fix_cmake_global_defaults(rmm::rmm)
    _fix_cmake_global_defaults(rmm::Thrust)
    _fix_cmake_global_defaults(rmm::spdlog_header_only)

    set(rmm_VERSION "${rmm_VERSION}" PARENT_SCOPE)
endfunction()

find_and_configure_rmm()
