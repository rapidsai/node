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

function(find_and_configure_cudf VERSION)

    include(get_cpm)

    execute_process(COMMAND node -p
                    "require('@nvidia/rapids-core').cpm_source_cache_path"
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    OUTPUT_VARIABLE NVIDIA_CPM_SOURCE_CACHE
                    OUTPUT_STRIP_TRAILING_WHITESPACE)

    set(CUDF_GENERATED_INCLUDE_DIR ${NVIDIA_CPM_SOURCE_CACHE}/cudf-build)

    CPMAddPackage(NAME  cudf
        VERSION         ${VERSION}
        # GIT_REPOSITORY https://github.com/rapidsai/cudf.git
        # GIT_TAG        branch-${VERSION}
        GIT_REPOSITORY  https://github.com/trxcllnt/cudf.git
        # Can also use a local path to your repo clone for testing
        # GIT_REPOSITORY  /home/ptaylor/dev/rapids/cudf
        GIT_TAG         fix/cmake-exports
        GIT_SHALLOW     TRUE
        SOURCE_SUBDIR   cpp
        OPTIONS         "BUILD_TESTS OFF"
                        "BUILD_BENCHMARKS OFF"
                        "ARROW_STATIC_LIB ON"
                        "JITIFY_USE_CACHE ON"
                        "CUDA_STATIC_RUNTIME ON"
                        "AUTO_DETECT_CUDA_ARCHITECTURES ON"
                        "DISABLE_DEPRECATION_WARNING ${DISABLE_DEPRECATION_WARNINGS}")

    # Because libcudf.so doesn't contain any symbols, the linker will determine
    # that it's okay to prune it before copying `DT_NEEDED` entries from it.
    set(CUDF_LIBRARY "-Wl,--no-as-needed" cudf::cudf "-Wl,--as-needed" PARENT_SCOPE)
endfunction()

find_and_configure_cudf(${CUDF_VERSION})
