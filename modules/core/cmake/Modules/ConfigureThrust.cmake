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

function(find_and_configure_thrust VERSION)
    _get_update_disconnected_state(Thrust ${VERSION} UPDATE_DISCONNECTED)
    CPMAddPackage(NAME  Thrust
        VERSION         ${VERSION}
        GIT_REPOSITORY  https://github.com/NVIDIA/thrust.git
        GIT_TAG         ${VERSION}
        GIT_SHALLOW     TRUE
        ${UPDATE_DISCONNECTED}
        PATCH_COMMAND   patch --reject-file=- -p1 -N < ${CMAKE_CURRENT_LIST_DIR}/thrust.patch || true
    )
endfunction()

find_and_configure_thrust(1.15.0)
