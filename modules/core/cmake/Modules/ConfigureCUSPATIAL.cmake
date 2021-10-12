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

    include(ConfigureCUDF)

    _clean_build_dirs_if_not_fully_built(cuspatial libcuspatial.so)

    _set_package_dir_if_exists(cuspatial cuspatial)

    if(NOT TARGET cuspatial::cuspatial)
        _get_major_minor_version(${VERSION} MAJOR_AND_MINOR)
        _get_update_disconnected_state(cuspatial ${VERSION} UPDATE_DISCONNECTED)
        CPMFindPackage(NAME     cuspatial
            VERSION             ${VERSION}
            GIT_REPOSITORY      https://github.com/rapidsai/cuspatial.git
            GIT_TAG             branch-${MAJOR_AND_MINOR}
            GIT_SHALLOW         TRUE
            ${UPDATE_DISCONNECTED}
            SOURCE_SUBDIR       cpp
            OPTIONS             "BUILD_TESTS OFF"
                                "BUILD_BENCHMARKS OFF"
                                "JITIFY_USE_CACHE ON"
                                "PER_THREAD_DEFAULT_STREAM ON"
                                "DISABLE_DEPRECATION_WARNING ON")
    endif()
    # Make sure consumers of our libs can see cuspatial::cuspatial
    _fix_cmake_global_defaults(cuspatial::cuspatial)
endfunction()

find_and_configure_cuspatial(${CUSPATIAL_VERSION})
