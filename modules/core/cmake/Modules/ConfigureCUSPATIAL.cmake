#=============================================================================
# Copyright (c) 2021, NVIDIA CORPORATION.
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

function(find_and_configure_cuspatial VERSION)

    include(get_cpm)

    CPMAddPackage(NAME  cuspatial
        VERSION         ${VERSION}
        # GIT_REPOSITORY https://github.com/rapidsai/cuspatial.git
        # GIT_TAG        branch-${VERSION}
        GIT_REPOSITORY  https://github.com/trxcllnt/cuspatial.git
        # Can also use a local path to your repo clone for testing
        # GIT_REPOSITORY  /home/ptaylor/dev/rapids/cuspatial
        GIT_TAG         combined-fixes
        GIT_SHALLOW     TRUE
        SOURCE_SUBDIR   cpp
        OPTIONS         "BUILD_TESTS OFF"
                        "BUILD_BENCHMARKS OFF"
                        "ARROW_STATIC_LIB ON"
                        "JITIFY_USE_CACHE ON"
                        "CUDA_STATIC_RUNTIME ON"
                        "PER_THREAD_DEFAULT_STREAM ON"
                        "DISABLE_DEPRECATION_WARNING ${DISABLE_DEPRECATION_WARNINGS}")
endfunction()

find_and_configure_cuspatial(${CUSPATIAL_VERSION})
