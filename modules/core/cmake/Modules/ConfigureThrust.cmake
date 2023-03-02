#=============================================================================
# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

function(find_and_configure_thrust VERSION)
    include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/get_cpm.cmake)
    _set_thrust_dir_if_exists()
    find_package(Thrust "${VERSION}.0" EXACT QUIET)
    if(NOT Thrust_FOUND)
      _get_update_disconnected_state(Thrust ${VERSION} UPDATE_DISCONNECTED)
      CPMAddPackage(NAME         Thrust
          VERSION                "${VERSION}.0"
          # EXCLUDE_FROM_ALL       TRUE
          GIT_REPOSITORY         https://github.com/NVIDIA/thrust.git
          GIT_TAG                ${VERSION}
          GIT_SHALLOW            TRUE
          ${UPDATE_DISCONNECTED}
          PATCH_COMMAND          patch --reject-file=- -p1 -N < ${CMAKE_CURRENT_LIST_DIR}/thrust.patch || true
      )
    endif()
    set(CPM_THRUST_CURRENT_VERSION "${VERSION}.0" CACHE STRING "version of thrust we checked out" FORCE)
endfunction()

find_and_configure_thrust(1.17.2)
