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

    execute_process(COMMAND node -p
                    "require('@rapidsai/core').cpm_source_cache_path"
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    OUTPUT_VARIABLE NODE_RAPIDS_CPM_SOURCE_CACHE
                    OUTPUT_STRIP_TRAILING_WHITESPACE)

    set(CUDF_GENERATED_INCLUDE_DIR ${NODE_RAPIDS_CPM_SOURCE_CACHE}/cudf-build)

    CPMAddPackage(NAME  cuspatial
        VERSION         ${VERSION}
        # GIT_REPOSITORY https://github.com/rapidsai/cuspatial.git
        # GIT_TAG        branch-${VERSION}
        GIT_REPOSITORY  https://github.com/trxcllnt/cuspatial.git
        # Can also use a local path to your repo clone for testing
        # GIT_REPOSITORY  /home/ptaylor/dev/rapids/cuspatial
        GIT_TAG         fix/cmake-exports
        GIT_SHALLOW     TRUE
        SOURCE_SUBDIR   cpp
        OPTIONS         "BUILD_TESTS OFF"
                        "BUILD_BENCHMARKS OFF"
                        "JITIFY_USE_CACHE ON"
                        "CUDA_STATIC_RUNTIME ON"
                        "CUDF_USE_ARROW_STATIC ON"
                        "PER_THREAD_DEFAULT_STREAM ON"
                        "DISABLE_DEPRECATION_WARNING ${DISABLE_DEPRECATION_WARNINGS}")

    # Make sure consumers of our libs can see cuspatial::cuspatial
    fix_cmake_global_defaults(cuspatial::cuspatial)

    if(NOT cuspatial_BINARY_DIR IN_LIST CMAKE_PREFIX_PATH)
        list(APPEND CMAKE_PREFIX_PATH "${cuspatial_BINARY_DIR}")
        set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} PARENT_SCOPE)
    endif()
endfunction()

find_and_configure_cuspatial(${CUSPATIAL_VERSION})
