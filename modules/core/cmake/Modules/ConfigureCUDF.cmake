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

function(find_and_configure_cudf VERSION)

    include(get_cpm)

    execute_process(COMMAND node -p
                    "require('@nvidia/rapids-core').cpm_source_cache_path"
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    OUTPUT_VARIABLE NODE_RAPIDS_CPM_SOURCE_CACHE
                    OUTPUT_STRIP_TRAILING_WHITESPACE)

    set(CUDF_GENERATED_INCLUDE_DIR ${NODE_RAPIDS_CPM_SOURCE_CACHE}/cudf-build)

    CPMAddPackage(NAME  cudf
        VERSION         ${VERSION}
        GIT_REPOSITORY  https://github.com/rapidsai/cudf.git
        GIT_TAG         branch-${VERSION}
        GIT_SHALLOW     TRUE
        SOURCE_SUBDIR   cpp
        OPTIONS         "BUILD_TESTS OFF"
                        "BUILD_BENCHMARKS OFF"
                        "JITIFY_USE_CACHE ON"
                        "CUDA_STATIC_RUNTIME ON"
                        "CUDF_USE_ARROW_STATIC ON"
                        "PER_THREAD_DEFAULT_STREAM ON"
                        "DISABLE_DEPRECATION_WARNING ${DISABLE_DEPRECATION_WARNINGS}")

    # Make sure consumers of our libs can also see cudf::cudf
    if(TARGET cudf::cudf)
        get_target_property(cudf_is_imported cudf::cudf IMPORTED)
        if(cudf_is_imported)
            set_target_properties(cudf::cudf PROPERTIES IMPORTED_GLOBAL TRUE)
        endif()
    endif()
endfunction()

find_and_configure_cudf(${CUDF_VERSION})
