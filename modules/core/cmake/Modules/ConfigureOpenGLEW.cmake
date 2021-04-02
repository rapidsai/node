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

if(GLEW_USE_STATIC_LIBS)
    set(GLEW_LIBRARY libglew_static)
else()
    set(GLEW_LIBRARY libglew_shared)
endif(GLEW_USE_STATIC_LIBS)

_set_package_dir_if_exists(${GLEW_LIBRARY} glew)

add_compile_definitions(GLEW_EGL)

function(find_and_configure_glew VERSION)
    if(NOT TARGET ${GLEW_LIBRARY})
        CPMFindPackage(NAME glew
            VERSION         ${VERSION}
            GIT_REPOSITORY  https://github.com/Perlmint/glew-cmake.git
            GIT_TAG         glew-cmake-${VERSION}
            GIT_SHALLOW     TRUE
            GIT_CONFIG      "advice.detachedhead=false"
            OPTIONS         "ONLY_LIBS 0"
                            # Ignore glew's missing VERSION
                            "CMAKE_POLICY_DEFAULT_CMP0048 NEW"
                            "glew-cmake_BUILD_MULTI_CONTEXT OFF"
                            "glew-cmake_BUILD_SINGLE_CONTEXT ON"
                            "glew-cmake_BUILD_SHARED ${GLEW_USE_SHARED_LIBS}"
                            "glew-cmake_BUILD_STATIC ${GLEW_USE_STATIC_LIBS}"
        )
    endif()
endfunction()

find_and_configure_glew(${GLEW_VERSION})
