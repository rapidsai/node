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

include(get_cpm)

CPMAddPackage(NAME rmm
    VERSION        ${RMM_VERSION}
    GIT_REPOSITORY https://github.com/rapidsai/rmm.git
    GIT_TAG        branch-${RMM_VERSION}
    OPTIONS        "BUILD_TESTS OFF"
                   "BUILD_BENCHMARKS OFF"
                   "DISABLE_DEPRECATION_WARNINGS ${DISABLE_DEPRECATION_WARNINGS}"
)

message(STATUS "rmm source dir: " ${rmm_SOURCE_DIR})

set(RMM_INCLUDE_DIR "${rmm_SOURCE_DIR}/include")
